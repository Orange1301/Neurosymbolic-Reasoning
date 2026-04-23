class DataFilter:
    '''
    A class used to filter First-Order Logic (FOL) predictions within a dataset entry.
    It deduplicates identical FOL strings and ensures logical syntax integrity by 
    checking for balanced brackets (crucial for catching truncated LLM outputs).
    
    Attributes
    ----------
    - duplicate_count : int
        Counter for the number of duplicate predictions removed.
    
    - syntax_error_count : int
        Counter for the number of predictions with invalid/unbalanced syntax.
        
    Methods
    -------
    - _is_valid_syntax(fol_str)
        Uses a stack-based approach to verify balanced parentheses, brackets, and braces.
        
    - filter(entry)
        Processes a single data entry, filtering its 'predictions' list.
    '''
    
    def __init__(self):
        self.duplicate_count = 0
        self.syntax_error_count = 0
    
    
    def _is_valid_syntax(self, fol_str: str) -> bool:
        '''
        Validates that all opening brackets/parentheses have corresponding closing pairs.
        
        Parameters
        ----------
        - fol_str : str
            The FOL string to validate.
            
        Returns
        -------
        - bool
            True if syntax is balanced, False otherwise.
        '''
        if not fol_str:
            return False
            
        stack = []
        matching_bracket = {')': '(', ']': '[', '}': '{'}
        
        for char in fol_str:                   
            if char in matching_bracket.values():
                stack.append(char)
            elif char in matching_bracket.keys():
                if not stack or stack.pop() != matching_bracket[char]:
                    return False

        return len(stack) == 0
    
    
    def filter(self, entry: dict) -> dict:
        '''
        Filters the 'predictions' list of a single entry by removing duplicates 
        and syntactically broken FOL strings.
        
        Parameters
        ----------
        - entry : dict
            A data point containing a "predictions" key with a list of FOL dicts.
            
        Returns
        -------
        - dict
            A copy of the entry containing only valid, unique predictions.
        '''
        seen_fols = set()
        valid_predictions = []
        
        for pred in entry.get("predictions", []):
            fol = pred.get("fol", "")
            
            # Check for duplicates within this specific entry
            if fol in seen_fols:
                self.duplicate_count += 1
                continue
            
            # Validate syntax (bracket balancing)
            if not self._is_valid_syntax(fol):
                print(fol)
                print('=' * 50)
                self.syntax_error_count += 1
                continue
            
            seen_fols.add(fol)
            valid_predictions.append(pred)
            
        filtered_element = entry.copy()
        filtered_element["predictions"] = valid_predictions
        
        return filtered_element