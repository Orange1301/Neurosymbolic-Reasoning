from sklearn.model_selection import train_test_split
from datasets import load_dataset
from model.DataFilter import DataFilter
import json
import os

dataset = load_dataset("yale-nlp/FOLIO")

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
        natural = data["premises"] + "" + data["conclusion"]        
        fol = (data["premises-FOL"] + "" + data["conclusion-FOL"])
        processed_data.append({
            "story_id": data["story_id"],
            "natural": natural,
            "fol": fol,
            "label": data["label"]
        })

    return processed_data

def save_data_list_to_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

SAVE_DIR = './output/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)
    
TEST_SAVE_PATH = os.path.join(SAVE_DIR, 'folio_test.json')
VALID_SAVE_PATH = os.path.join(SAVE_DIR, 'folio_valid.json')
TRAIN_SAVE_PATH = os.path.join(SAVE_DIR, 'folio_train.json')

processed_examined_data = preprocess(dataset['validation'].to_list())
story_ids = set([d["story_id"] for d in processed_examined_data])
valid_ids, test_ids = train_test_split(list(story_ids), test_size=0.5)

processed_test_data = [d for d in processed_examined_data if d["story_id"] in test_ids]
processed_valid_data = [d for d in processed_examined_data if d["story_id"] in valid_ids]
processed_train_data = preprocess(dataset['train'].to_list())

# NOTICE: Always filter with this order (Test -> Valid -> Train)
logic_filter = DataFilter()
final_test_data = logic_filter.filter_list(processed_test_data)
final_valid_data = logic_filter.filter_list(processed_valid_data)
final_train_data = logic_filter.filter_list(processed_train_data)

save_data_list_to_json(final_test_data, TEST_SAVE_PATH)
save_data_list_to_json(final_valid_data, VALID_SAVE_PATH)
save_data_list_to_json(final_train_data, TRAIN_SAVE_PATH)

print('Before filtering:')
print(f'- Train: {len(processed_train_data)}')
print(f'- Valid: {len(processed_valid_data)}')
print(f'- Test: {len(processed_test_data)}')
print('=' * 20)
print('After filtering:')
print(f'- Train: {len(final_train_data)}')
print(f'- Valid: {len(final_valid_data)}')
print(f'- Test: {len(final_test_data)}')