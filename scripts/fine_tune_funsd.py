import argparse
import numpy as np
import wandb

from datasets import Features, Sequence, Value, Array2D, Array3D, load_dataset
from evaluate import load as load_metric
from transformers import AutoProcessor, Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator
from superdoc import LayoutLMv3Config, LayoutLMv3ForTokenClassification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_path", type=str, default="albertklorer/layoutlmv3-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    wandb.init(project="layoutlmv3-funsd-fine-tune", config=vars(args))

    dataset = load_dataset("nielsr/funsd-layoutlmv3")
    label_list = dataset["train"].features["ner_tags"].feature.names
    id2label = {
        i: l for
        i, l in
        enumerate(label_list)
    }
    label2id = {
        l: i
        for i, l
        in enumerate(label_list)
    }

    config = LayoutLMv3Config.from_pretrained(args.hf_path, num_labels=len(label_list), id2label=id2label, label2id=label2id)
    processor = AutoProcessor.from_pretrained(args.hf_path, apply_ocr=False)

    def prepare_examples(examples):
        return processor(
            examples["image"],
            examples["tokens"],
            boxes=examples["bboxes"],
            word_labels=examples["ner_tags"],
            truncation=True,
            padding="max_length",
        )
    
    img_size = config.input_size
    seq_len = config.max_position_embeddings - 2

    features = Features({
        "pixel_values": Array3D(dtype="float32", shape=(config.num_channels, img_size, img_size)),
        "input_ids": Sequence(feature=Value(dtype="int64")),
        "attention_mask": Sequence(Value(dtype="int64")),
        "bbox": Array2D(dtype="int64", shape=(seq_len, 4)),
        "labels": Sequence(feature=Value(dtype="int64")),
    })

    train_dataset = dataset["train"].map(prepare_examples, batched=True, remove_columns=dataset["train"].column_names, features=features)
    eval_dataset = dataset["test"].map(prepare_examples, batched=True, remove_columns=dataset["test"].column_names, features=features)

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    model = LayoutLMv3ForTokenClassification.from_pretrained(args.hf_path, config=config)

    metric = load_metric("seqeval")

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=2)
        labels = eval_pred.label_ids
        true_preds = [
            [
                label_list[p]
                for p, l in
                zip(pred, label)
                if l != -100
            ]
            for pred, label
            in zip(preds, labels)
        ]
        true_labels = [
            [
                label_list[l]
                for _, l in
                zip(pred, label)
                if l != -100
            ]
            for pred, label
            in zip(preds, labels)
        ]

        # Calculate token-level accuracy
        correct = 0
        total = 0
        for pred, label in zip(preds, labels):
            for p, l in zip(pred, label):
                if l != -100:
                    total += 1
                    if p == l:
                        correct += 1
        accuracy = correct / total if total > 0 else 0.0

        results = {
            "f1": metric.compute(predictions=true_preds, references=true_labels)["overall_f1"],
            "accuracy": accuracy,
        }

        return results

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./output",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            eval_strategy="epoch",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=1,
            report_to="wandb",
            seed=args.seed,
            remove_unused_columns=False,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
