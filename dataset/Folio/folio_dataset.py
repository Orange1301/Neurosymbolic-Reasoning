from datasets import load_dataset
import json
from nltk.sem.logic import LogicParser
import re

dataset = load_dataset("yale-nlp/FOLIO")

# def fol_translate(fol_text: str):
#     replacements = {
#         '∀': 'all ', 
#         '∃': 'exists ',
#         '∧': '&', 
#         '∨': '|',
#         '⊕': '^',
#         '¬': '-',
#         '→': '->', 
#         '⟷': '<->'
#     }
#     for k, v in replacements.items():
#         fol_text = fol_text.replace(k, v)

#     # Add dot: "all x (P(x))" --> "all x. (P(x))"
#     fol_text = re.sub(r'(all|exists)\s+([a-z0-9]+)\s*', r'\1 \2. ', fol_text)
#     return fol_text

# def is_valid_fol(fol_list):
#     parser = LogicParser()
#     try:
#         for line in fol_list:
#             if line.strip():
#                 parser.parse(line)
#         return True
#     except Exception:
#         return False

def preprocess(data_list):
    '''
        Output: list
        [
            {
                "story_id": "...",
                "natural": "...",
                "fol": "..."
                "label": "..."
            }
        ]
    '''
    processed_data = []
    for data in data_list:
        natural = data["premises"] + "\n" + data["conclusion"]
        
        fol = (data["premises-FOL"] + "\n" + data["conclusion-FOL"])


        # raw_fol_text = data["premises-FOL"] + "\n" + data["conclusion-FOL"]
        # # translated_fol_text = fol_translate(raw_fol_text)
        # fol = translated_fol_text.split('\n')

        # if is_valid_fol(fol):
        processed_data.append({
            "story_id": data["story_id"],
            "natural": natural,
            "fol": fol,
            "label": data["label"]
        })

    return processed_data

# for split in dataset.keys():
#     save_path = f"folio_{split}.json"
#     data_list = dataset[split].to_list()

#     processed_data = preprocess(data_list)
    
#     with open(save_path, "w", encoding="utf-8") as f:
#         json.dump(processed_data, f, ensure_ascii=False, indent=4)

# Train dataset ----
train_save_path = f"folio_train.json"
train_data_list = dataset['train'].to_list()

processed_train_data = preprocess(train_data_list)

with open(train_save_path, "w", encoding="utf-8") as f:
    json.dump(processed_train_data, f, ensure_ascii=False, indent=4)

# Valid and Test dataset ----
from sklearn.model_selection import train_test_split
valid_save_path = "folio_valid.json"
test_save_path = "folio_test.json"
processed_data = preprocess(dataset['validation'].to_list())

story_ids = set([d["story_id"] for d in processed_data])
valid_ids, test_ids = train_test_split(list(story_ids), test_size=0.5)
print(valid_ids)
print(test_ids)
processed_valid_data = [d for d in processed_data if d["story_id"] in valid_ids]
processed_test_data = [d for d in processed_data if d["story_id"] in test_ids]

with open(valid_save_path, "w", encoding="utf-8") as f:
    json.dump(processed_valid_data, f, ensure_ascii=False, indent=4)
with open(test_save_path, "w", encoding="utf-8") as f:
    json.dump(processed_test_data, f, ensure_ascii=False, indent=4)