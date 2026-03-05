import nltk
from nltk.sem.logic import LogicParser
from nltk.inference import Prover9
import regex as re

parser = LogicParser()
prover9_path = 'C:\\Program Files (x86)\\Prover9-Mace4\\bin-win32' 
prover = Prover9()
prover.config_prover9(prover9_path)

def translate_fol(fol_text: str):
    # '-' --> '_'
    fol_text = re.sub(r'(?<=[a-zA-Z0-9])-(?=[a-zA-Z0-9])', '_', fol_text)

    replacements = {
        '∀': 'all ', 
        '∃': 'exists ',
        '∧': '&', 
        '∨': '|',
        '⊕': '^',
        '¬': '-',
        '→': '->', 
        '⟷': '<->'
    }
    for k, v in replacements.items():
        fol_text = fol_text.replace(k, v)

    # Add dot: "all x (P(x))" --> "all x. (P(x))"
    fol_text = re.sub(r'(all|exists)\s+([a-z0-9]+)\s*', r'\1 \2. ', fol_text)

    # Fix prover9 constants name (eg: "yuri" -> "c_yuri")
    words = re.findall(r'\b[a-z][a-zA-Z0-9_]*\b', fol_text)
    reserved_words = {'all', 'exists', 'u', 'v', 'w', 'x', 'y', 'z'}
    
    for w in set(words):
        if w not in reserved_words:
            fol_text = re.sub(fr'\b{w}\b', f'c_{w}', fol_text)
    return fol_text

def is_valid_fol(fol_list):
    try:
        for line in fol_list:
            if line.strip():
                parser.parse(line)
        return True
    except Exception:
        return False

def check_conclusion(premises_list, conclusion_str):
    # Read fol strings
    parser = LogicParser()
    translated_premises = [translate_fol(p) for p in premises_list]
    translated_conclusion = translate_fol(conclusion_str)

    # if (not is_valid_fol(translated_premises) or not is_valid_fol([translated_conclusion])):
    #     error_msg = f"Invalid FOL syntax detected!"
    #     raise ValueError(error_msg)
    
    try:
        parsed_premises = [parser.parse(p) for p in translated_premises]
        parsed_conclusion = parser.parse(translated_conclusion)
    except Exception as e:
        print(e)
        raise f"Error: {e}"

    # Check conclusion
    is_true = prover.prove(parsed_conclusion, parsed_premises)
    if is_true:
        return "True"

    is_false = prover.prove(parsed_conclusion.negate(), parsed_premises)
    if is_false:
        return "False"

    return "Uncertain"

# Quick test ------------------------
# premises = [
#     "∀x (InThisClub(x) ∧ PerformOftenIn(x, schoolTalentShow) → Attend(x, schoolEvent) ∧ VeryEngagedWith(x, schoolEvent))",
#     "InThisClub(bonnie) ∧ PerformOftenIn(bonnie, schoolTalentShow)" 
# ]
# conclusion = "Attend(bonnie, schoolEvent)"

# print(f"Result: {check_fol_string(premises, conclusion)}")

# Test on mock_data.json ----------------
import json
with open('mock_data.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

wrong_predictions = []
total = 0
count_correct = 0
count_wrong = 0
count_error = 0
for data in test_data:
    total += 1
    try:
        predicted = check_conclusion(data["fol"][:-1], data["fol"][-1])
        label = data["label"]

        if (predicted != label):
            count_wrong += 1
            # wrong_predictions.append((data["story_id"], predicted, label))
            print((data["story_id"], predicted, label))
        else:
            # print("Correct: ", label)
            count_correct += 1
    except Exception:
        count_error += 1

print("Total: ", total)
print("Correct: ", count_correct)
print("Wrong: ", count_wrong)
print("Error:", count_error)


