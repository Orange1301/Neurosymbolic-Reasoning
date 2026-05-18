from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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


def normalize_example(item: Dict[str, Any], index: int) -> Dict[str, str]:
    return {
        "id": as_text(item.get("id") or item.get("story_id") or f"example_{index}"),
        "nat_premises": as_text(first_present(item, ("nat_premises", "nat_premise", "premises"))),
        "nat_conclusion": as_text(first_present(item, ("nat_conclusion", "conclusion"))),
        "fol_premises": as_text(first_present(item, ("fol_premises", "fol_premise"))),
        "fol_conclusion": as_text(item.get("fol_conclusion")),
        "label": as_text(item.get("label")),
    }


def load_examples(path: str, max_samples: Optional[int]) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    examples = [normalize_example(item, index) for index, item in enumerate(raw)]
    return examples[:max_samples] if max_samples is not None else examples


def build_source(example: Dict[str, str]) -> str:
    return (
        "translate natural language to first-order logic:\n"
        "Premises:\n"
        f"{example['nat_premises']}\n"
        "Conclusion:\n"
        f"{example['nat_conclusion']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FOL samples from a Basic 2 mT5 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid-file", required=True)
    parser.add_argument(
        "--output-file", "--output", dest="output_file", default="outputs/basic2_mt5_folio/sample_generations.json"
    )
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--max-source-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.num_examples is not None:
        args.max_samples = args.num_examples
    return args


def main() -> int:
    args = parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.decoder_start_token_id is None and model.config.pad_token_id is not None:
        model.config.decoder_start_token_id = model.config.pad_token_id

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)
    model.eval()

    rows = []
    for example in load_examples(args.valid_file, args.max_samples):
        source = build_source(example)
        inputs = tokenizer(source, return_tensors="pt", truncation=True, max_length=args.max_source_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, num_beams=1, pad_token_id=tokenizer.pad_token_id
            )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        rows.append(
            {
                "id": example["id"],
                "input": source,
                "gold_fol": f"{example['fol_premises']}\n{example['fol_conclusion']}".strip(),
                "generated_fol": generated,
            }
        )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved sample generations to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
