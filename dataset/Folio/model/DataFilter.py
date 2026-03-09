import hashlib
import re

class DataFilter:
    '''
    A class used to filter the FOL dataset by applying logical uniqueness, preventing label leakage, and verifying that symbolic constants in the logic accurately reflect the natural language narrative.
    
    Attributes
    ----------
    - _global_seen_signatures : set
        A persistent registry of MD5 hashes representing unique logic-label pairs 
        encountered across all filtered lists.
    
    - NATURAL_COL : str
        The natural key's name
    
    - FOL_COL : str
        The FOL key's name
        
    - LABEL_COL : str
        The label key's name
        
    - MIN_LINES_OF_FOL : int
        The minimun number lines of statement
        
    Methods
    -------
    - _generate_signature(entry)
        Generates a deterministic MD5 hash of normalized FOL strings and labels.
        
    - _is_structural_validation(entry)
        Validates logical integrity, including minimum premise count, and premise-conclusion overlap (leakage)
        
    - filter_list(data_list)
        Filters a list of dictionaries against structural rules and previously seen signatures to maintain split integrity.
    '''
    
    def __init__(self):
        self._global_seen_signatures = set()
        self.NATURAL_COL = 'natural'
        self.FOL_COL = 'fol'
        self.LABEL_COL = 'label'
        self.MIN_LINES_OF_FOL = 2
        
        
    def _generate_signature(self, entry:dict) -> str:
        '''
        Generates a deterministic MD5 hash of normalized FOL strings and labels.
        
        This ensures that entries with identical logic but different whitespace, line breaks, or casing are treated as duplicates.
        
        Parameters
        ----------
        - entry : dict
            A single data point containing FOL and label fields.
            
        Returns
        -------
        str
            A hexadecimal MD5 hash representing the unique logic-label signature.
        '''

        # Normalize FOL: remove all whitespaces and lowercase the text
        fol_clean = ''.join(entry.get(self.FOL_COL, '').lower().split())
        
        # Normalize label: lowercase and strip to ensure consistency
        label_clean = str(entry.get(self.LABEL_COL, '')).lower().strip()
        
        combined_str = f'{fol_clean}|{label_clean}'
        return hashlib.md5(combined_str.encode()).hexdigest()
    
    
    def _is_structural_validation(self, entry:dict) -> bool:
        '''
        Validates logical integrity, including minimum premise count, and premise-conclusion overlap (leakage)
        
        This ensures that entries mismatching the requirements are treated as invalid entries.
        
        Parameters
        ----------
        - entry : dict
            A single data point containing FOL and label fields.
            
        Returns
        -------
        bool
            True if the entry passes all structural requirements, False otherwise.
        '''
        
        # Get natural and fol text
        natural_text = entry.get(self.NATURAL_COL, '').lower()
        fol_text = entry.get(self.FOL_COL, '')
        
        # Requirement 1: Check if the entry has the minimum statement lines
        fol_lines = [line.strip() for line in fol_text.split('\n') if line.strip()]
        if len(fol_lines) < self.MIN_LINES_OF_FOL:
            return False
        
        # Requirement 2: Data leakage (check if the conclusion is already a premise)
        premises = fol_lines[:-1]
        conclusion = fol_lines[-1]
        if conclusion in premises:
            return False
                
        return True
    
    def filter_list(self, data_list):
        '''
        Filters a list of dictionaries against structural rules and previously seen signatures to maintain split integrity.
        
        Parameters
        ----------
        - data_list : list[dict]
            A collection of data points, each containing story ID, Natural text, FOL, and Label.
            
        Returns
        -------
        list[dict]
            A subset of the input list containing only unique, structurally valid entries.
        '''
        
        unique_results = []
        
        for entry in data_list:
            signature = self._generate_signature(entry)
            
            if signature not in self._global_seen_signatures:
                if self._is_structural_validation(entry):
                    unique_results.append(entry)
                    self._global_seen_signatures.add(signature)
                
        return unique_results