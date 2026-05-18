from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class Example:
    id: str
    nat_premises: str
    nat_conclusion: str
    fol_premises: str
    fol_conclusion: str
    label: str


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def first_present(item: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None


def normalize_example(item: Dict[str, Any], index: int) -> Example:
    example_id = as_text(item.get("id") or item.get("story_id") or f"example_{index}")
    example = Example(
        id=example_id,
        nat_premises=as_text(first_present(item, ("nat_premises", "nat_premise", "premises"))),
        nat_conclusion=as_text(first_present(item, ("nat_conclusion", "conclusion"))),
        fol_premises=as_text(first_present(item, ("fol_premises", "fol_premise"))),
        fol_conclusion=as_text(item.get("fol_conclusion")),
        label=as_text(item.get("label")),
    )
    missing = [
        field
        for field in ("nat_premises", "nat_conclusion", "fol_premises", "fol_conclusion", "label")
        if not getattr(example, field)
    ]
    if missing:
        raise ValueError(f"Example {example_id!r} is missing required field(s): {', '.join(missing)}")
    return example


def load_examples(path: str, max_samples: Optional[int] = None) -> List[Example]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list.")
    examples = [normalize_example(item, i) for i, item in enumerate(raw)]
    if max_samples is not None:
        examples = examples[:max_samples]
    return examples


def build_source(example: Example) -> str:
    return (
        "translate natural language to first-order logic:\n"
        "Premises:\n"
        f"{example.nat_premises}\n"
        "Conclusion:\n"
        f"{example.nat_conclusion}"
    )


def build_target(example: Example) -> str:
    return f"{example.fol_premises}\n{example.fol_conclusion}".strip()


def make_rows(examples: Sequence[Example]) -> List[Dict[str, str]]:
    return [{"id": ex.id, "source": build_source(ex), "target": build_target(ex)} for ex in examples]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mT5 for Basic 2: natural language -> FOL on FOLIO.")
    parser.add_argument("--profile", choices=("smoke", "base", "large"), default="base")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--valid-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--max-source-length", type=int)
    parser.add_argument("--max-target-length", type=int)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--per-device-eval-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument("--optim")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--predict-with-generate", action="store_true")
    parser.add_argument("--generation-max-length", type=int)
    args = parser.parse_args()
    apply_profile_defaults(args)
    return args


def apply_profile_defaults(args: argparse.Namespace) -> None:
    profiles: Dict[str, Dict[str, Any]] = {
        "smoke": {
            "model_name": "google/mt5-small",
            "max_source_length": 256,
            "max_target_length": 256,
            "num_train_epochs": 1,
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "eval_steps": 50,
            "save_steps": 50,
            "logging_steps": 5,
            "save_total_limit": 2,
            "optim": "adafactor",
            "generation_max_length": 256,
        },
        "base": {
            "model_name": "google/mt5-base",
            "max_source_length": 768,
            "max_target_length": 768,
            "num_train_epochs": 20,
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "eval_steps": 25,
            "save_steps": 25,
            "logging_steps": 5,
            "save_total_limit": 3,
            "optim": "adafactor",
            "generation_max_length": 768,
        },
        "large": {
            "model_name": "google/mt5-large",
            "max_source_length": 512,
            "max_target_length": 512,
            "num_train_epochs": 10,
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "eval_steps": 25,
            "save_steps": 25,
            "logging_steps": 5,
            "save_total_limit": 2,
            "optim": "adafactor",
            "generation_max_length": 512,
        },
    }
    defaults = profiles[args.profile]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.profile in {"base", "large"}:
        args.gradient_checkpointing = True if not args.gradient_checkpointing else args.gradient_checkpointing
        args.predict_with_generate = True if not args.predict_with_generate else args.predict_with_generate


def training_args_kwargs(args: argparse.Namespace, has_cuda: bool, has_eval: bool) -> Dict[str, Any]:
    import inspect
    from transformers import Seq2SeqTrainingArguments

    profile_wants_fp16 = args.profile in {"base", "large"}
    fp16 = bool((args.fp16 or profile_wants_fp16) and has_cuda)
    bf16 = bool(args.bf16 and has_cuda)
    if args.fp16 and not has_cuda:
        print("WARNING: --fp16 requested but CUDA is unavailable; disabling fp16.")
    if args.bf16 and not has_cuda:
        print("WARNING: --bf16 requested but CUDA is unavailable; disabling bf16.")
    if fp16 and bf16:
        print("WARNING: both --fp16 and --bf16 requested; using bf16 and disabling fp16.")
        fp16 = False

    candidates: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": "none",
        "seed": args.seed,
        "predict_with_generate": args.predict_with_generate,
        "generation_max_length": args.generation_max_length,
        "fp16": fp16,
        "bf16": bf16,
        "load_best_model_at_end": has_eval,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_strategy": "steps",
        "optim": args.optim,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataloader_pin_memory": False,
    }

    signature = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        candidates["eval_strategy"] = "steps" if has_eval else "no"
    elif "evaluation_strategy" in signature:
        candidates["evaluation_strategy"] = "steps" if has_eval else "no"

    return {key: value for key, value in candidates.items() if key in signature}


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(f"Distributed rank={rank}, local_rank={local_rank}, world_size={world_size}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"CUDA device {index}: {torch.cuda.get_device_name(index)}")

    train_examples = load_examples(args.train_file, args.max_train_samples)
    valid_examples = load_examples(args.valid_file, args.max_eval_samples)
    print(f"Loaded train examples: {len(train_examples)} from {args.train_file}")
    print(f"Loaded valid examples: {len(valid_examples)} from {args.valid_file}")

    train_rows = make_rows(train_examples)
    valid_rows = make_rows(valid_examples)
    if not train_rows:
        raise ValueError("No training examples loaded.")
    if not valid_rows:
        raise ValueError("No validation examples loaded.")

    print("Loading tokenizer:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("Loading model:", args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.decoder_start_token_id is None and model.config.pad_token_id is not None:
        model.config.decoder_start_token_id = model.config.pad_token_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        model_inputs = tokenizer(batch["source"], max_length=args.max_source_length, truncation=True)
        try:
            labels = tokenizer(text_target=batch["target"], max_length=args.max_target_length, truncation=True)
        except TypeError:
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(batch["target"], max_length=args.max_target_length, truncation=True)
        pad_id = tokenizer.pad_token_id
        label_ids = labels["input_ids"]
        if pad_id is not None:
            label_ids = [[token if token != pad_id else -100 for token in seq] for seq in label_ids]
        model_inputs["labels"] = label_ids
        return model_inputs

    train_dataset = Dataset.from_list(train_rows).map(tokenize_batch, batched=True, remove_columns=["id", "source", "target"])
    eval_dataset = Dataset.from_list(valid_rows).map(tokenize_batch, batched=True, remove_columns=["id", "source", "target"])

    first_labels = train_dataset[0]["labels"]
    non_ignored = sum(1 for token in first_labels if token != -100)
    if non_ignored == 0:
        raise ValueError("First tokenized training example has labels all set to -100.")
    print(f"First training example label tokens not ignored: {non_ignored}")

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)
    has_cuda = bool(torch.cuda.is_available())
    train_args = training_args_kwargs(args, has_cuda=has_cuda, has_eval=True)
    print("TrainingArguments:", json.dumps({k: str(v) for k, v in train_args.items()}, indent=2))

    trainer = Seq2SeqTrainer(
        model=model,
        args=__import__("transformers").Seq2SeqTrainingArguments(**train_args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if trainer.is_world_process_zero():
        with open(Path(args.output_dir) / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": args.model_name,
                    "train_file": args.train_file,
                    "valid_file": args.valid_file,
                    "num_train_examples": len(train_examples),
                    "num_valid_examples": len(valid_examples),
                    "args": vars(args),
                    "final_eval_metrics": metrics,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Saved final model/tokenizer to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
