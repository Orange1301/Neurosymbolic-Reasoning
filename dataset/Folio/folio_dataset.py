from datasets import load_dataset
import json

dataset = load_dataset("yale-nlp/FOLIO")

def processing1(data_list):
    '''
        Output: list
        [
            {
                "story_id": "...",
                "natural": "...",
                "fol": "..."
            }
        ]
    '''
    processed_data = []
    for data in data_list:
        natural = data["premises"] + "\n" + data["conclusion"]
        fol = data["premises-FOL"] + "\n" + data["conclusion-FOL"]
        processed_data.append({
            "story_id": data["story_id"],
            "natural": natural,
            "fol": fol
        })
    return processed_data

def processing2(data_list):
    '''
        Output: list
        [
            {
                "story_id": "...",
                "natural": ["...", "..."],
                "fol": ["...", "..."]
            }
        ]
    '''
    processed_data = []
    for data in data_list:
        natural = data["premises"].split('\n')
        natural.append(data["conclusion"])

        fol = data["premises-FOL"].split('\n')
        fol.append(data["conclusion-FOL"])

        processed_data.append({
            "story_id": data["story_id"],
            "natural": natural,
            "fol": fol
        })
    return processed_data

for split in dataset.keys():
    save_path = f"folio_{split}.json"
    data_list = dataset[split].to_list()

    processed_data = processing2(data_list)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
