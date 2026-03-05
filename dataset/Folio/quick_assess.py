import json

union_set = set()
# valid ---
with open('folio_valid.json', 'r', encoding='utf-8') as f:
    valid_data = json.load(f)
valid_ids = set()
for d in valid_data:
    valid_ids.add(d["story_id"])
    union_set.add(d["story_id"])

# test ---
with open('folio_test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
test_ids = set()
for d in test_data:
    test_ids.add(d["story_id"])
    union_set.add(d["story_id"])

# print out
print("valid:", sorted(valid_ids))
print("test:", sorted(test_ids))

print("Count shared stories:", len(valid_ids) + len(test_ids) - len(union_set))