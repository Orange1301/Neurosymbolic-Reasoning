from typing import Optional

def filter2(samples: list[dict],
            ground_truth: str,
            threshold: float = 0.5) -> Optional[tuple[list[dict], list[dict]]]:
    count = {"True": 0, "False": 0, "Uncertain": 0}
    for sample in samples:
        count[sample["label"]] += 1
        
    if count[ground_truth] / len(samples) > threshold:
        positive_sample = [sample for sample in samples if sample["label"] == ground_truth]
        negative_sample = [sample for sample in samples if sample["label"] != ground_truth]
        return positive_sample, negative_sample

    return None