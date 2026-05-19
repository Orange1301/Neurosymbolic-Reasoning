import nltk
from nltk.sem.logic import LogicParser
from nltk.inference import Prover9
import regex as re  

class Engine:
    def __init__(self, prover9_path='C:\\Program Files (x86)\\Prover9-Mace4\\bin-win32'):
        self.parser = LogicParser()
        self.prover9_path = prover9_path
        self.prover = Prover9()
        self.prover.config_prover9(prover9_path)

    def _transform_xor(self, text):
        """
        Transform A ⊕ B to ¬(A ⟷ B)
        """
        while '⊕' in text:
            idx = text.find('⊕')
            
            # Find left component
            lhs_end = idx
            while lhs_end > 0 and text[lhs_end-1].isspace():
                lhs_end -= 1
                
            curr = lhs_end - 1
    
            if text[curr] == ')':
                balance = 0
                while curr >= 0:
                    if text[curr] == ')': balance += 1
                    elif text[curr] == '(': balance -= 1
                    curr -= 1
                    if balance == 0: break
                while curr >= 0 and (text[curr].isalnum() or text[curr] == '_'):
                    curr -= 1
                lhs_start = curr + 1
            else:
                while curr >= 0 and (text[curr].isalnum() or text[curr] == '_'):
                    curr -= 1
                lhs_start = curr + 1
                
            lhs = text[lhs_start:lhs_end]
    
            # Find right component
            rhs_start = idx + 1
            while rhs_start < len(text) and text[rhs_start].isspace():
                rhs_start += 1
                
            curr = rhs_start
            if text[curr].isalnum() or text[curr] == '_':
                while curr < len(text) and (text[curr].isalnum() or text[curr] == '_'):
                    curr += 1
                if curr < len(text) and text[curr] == '(':
                    balance = 0
                    while curr < len(text):
                        if text[curr] == '(': balance += 1
                        elif text[curr] == ')': balance -= 1
                        curr += 1
                        if balance == 0: break
                rhs_end = curr
            elif text[curr] == '(':
                balance = 0
                while curr < len(text):
                    if text[curr] == '(': balance += 1
                    elif text[curr] == ')': balance -= 1
                    curr += 1
                    if balance == 0: break
                rhs_end = curr
            else:
                rhs_end = curr + 1
                
            rhs = text[rhs_start:rhs_end]
    
            # Transform XOR
            new_expression = f"¬({lhs} ⟷ {rhs})"
            text = text[:lhs_start] + new_expression + text[rhs_end:]
    
        return text
    
    def _translate_fol(self, fol_text: str):
        # Transform XOR
        fol_text = self._transform_xor(fol_text)
        
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
            '⟷': '<->',
            '↔': '<->',
            '≠': '!='
        }
        for k, v in replacements.items():
            fol_text = fol_text.replace(k, v)

        # Add dot: "all x (P(x))" --> "all x. (P(x))"
        fol_text = re.sub(r'(all|exists)\s+([a-z0-9]+)\s*', r'\1 \2. ', fol_text)

        # Fix prover9 constants name (eg: "yuri" -> "c_yuri")
        words = re.findall(r'\b[a-z][a-zA-Z0-9_]*\b', fol_text)
        reserved_words = {'all', 'exists', 'u', 'v', 'w', 'x', 'y', 'z'}
        
        for w in set(words):
            if w not in reserved_words and len(w) > 2:
                fol_text = re.sub(fr'\b{w}\b', f'c_{w}', fol_text)
        return fol_text


    def _is_valid_fol(self, fol_list):
        try:
            for line in fol_list:
                if line.strip():
                    self.parser.parse(line)
            return True
        except Exception as e:
            print(e)
            return False

    def _check_conclusion(self, premises, conclusion):
        '''
        This function is the engine. It checks whether the conclusion is True/False/Uncertain based on the list of premises.
        Args:
            premises: list[str]
            conclusion: str
        Returns: 
            str ("True"/"False"/"Uncertain")
        '''
        # Read fol strings
        translated_premises = [self._translate_fol(p) for p in premises]
        translated_conclusion = self._translate_fol(conclusion)

        if (not self._is_valid_fol(translated_premises) or not self._is_valid_fol([translated_conclusion])):
            error_msg = f"Invalid FOL syntax detected!"
            raise ValueError(error_msg)
        
        try:
            parsed_premises = [self.parser.parse(p) for p in translated_premises]
            parsed_conclusion = self.parser.parse(translated_conclusion)
        except Exception as e:
            raise f"Error: {e}"

        # Check conclusion
        if self.prover.prove(parsed_conclusion, []):
            raise Exception("Error: The conclusion itself is True!")
        elif self.prover.prove(parsed_conclusion.negate(), []):
            raise Exception("Error: The conclusion itself is False!")
        elif not self.prover.prove(None, parsed_premises):
            is_true = self.prover.prove(parsed_conclusion, parsed_premises)
            if is_true:
                return "True"

            is_false = self.prover.prove(parsed_conclusion.negate(), parsed_premises)
            if is_false:
                return "False"

            return "Uncertain"
        else: 
            raise Exception("Error: The premises are inconsistent!")
    
    def check_conclusion(self, data):
        fol_list = data["predictions"]
        new_fol_list = []
        error_list = []
        for fol in fol_list:
            try:
                premises = fol["fol"].split("\n")[:-1]
                conclusion = fol["fol"].split("\n")[-1]
                result = self._check_conclusion(premises, conclusion)

                if result == "True":
                    fol["label"] = "True"
                elif result == "False":   # FIX 2
                    fol["label"] = "False"
                else:
                    fol["label"] = "Uncertain"
                new_fol_list.append(fol)
            except Exception as e:
                error_list.append({
                    "premises": "\n".join(fol["fol"].split("\n")[:-1]),
                    "conclusion": fol["fol"].split("\n")[-1],
                    "error": repr(e)
                })
                continue
        data["predictions"] = new_fol_list
        return data, error_list
                

    
# def main():
#     # TEST ON 1 SAMPLE ------------------------
#     premises = [
#         "∀x (InThisClub(x) ∧ PerformOftenIn(x, schoolTalentShow) → Attend(x, schoolEvent) ∧ VeryEngagedWith(x, schoolEvent))",
#         "InThisClub(bonnie) ∧ PerformOftenIn(bonnie, schoolTalentShow)" 
#     ]
#     # conclusion = "Attend(bonnie, schoolEvent)"
#     conclusion = "¬InThisClub(bonnie) ∧ InThisClub(bonnie)"

#     print(f"Result: {check_conclusion(premises, conclusion)}")

# TEST ON MOCK DATA ----------------
import json

def main():
    with open('mock_data.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    engine = Engine()

    wrong_data = []; labels = {}
    total = 0
    count_correct = 0
    count_wrong = 0
    count_error = 0
    for id, fol in test_data.items():
        total += 1
        try:
            # fol_list = data["fol"].split('\n')
            # predicted = check_conclusion(fol_list[:-1], fol_list[-1])
            fol_list = fol.split('. ')
            premises, conclusion = fol_list[:-1], fol_list[-1]
            predicted = engine._check_conclusion(premises, conclusion)
            labels[id] = predicted
            # label = data["label"]

        #     if (predicted != label):
        #         count_wrong += 1
        #         wrong_data.append(data)
        #         print((data["id"], predicted, label))
        #     else:
        #         count_correct += 1
        except Exception as e:
            count_error += 1
            story_id = id
            wrong_data.append(id)
            print(f"{story_id}: {e}")

    # with open("wrong_folio_train.json", "w", encoding="utf-8") as f:
    #     json.dump(wrong_data, f, ensure_ascii=False, indent=4)
    # print([d["id"] for d in wrong_data])
    print(wrong_data)
    print(len(wrong_data))

    with open("predicted_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

    # print("Total: ", total)
    # print("Correct: ", count_correct)
    # print("Wrong: ", count_wrong)
    # print("Error:", count_error)

'''
For folio_train.json: 
- Total:  955
- Correct:  540
- Wrong:  373
- Error: 42

For folio_valid.json:
- Total:  97
- Correct:  68
- Wrong:  29
- Error: 0

For folio_test.json:
- Total:  97
- Correct:  68
- Wrong:  28
- Error: 1

FACT: The wrong answers and errors are not the engine's fault, it's because those samples' fols are incorrect.
'''

if __name__ == "__main__":
    main()