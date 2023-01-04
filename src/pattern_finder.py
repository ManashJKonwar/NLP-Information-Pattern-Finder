__author__ = "konwar.m"
__copyright__ = "Copyright 2023, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import pandas as pd

class PatternFinder:
    def __init__(self, data: pd.DataFrame, textual_col: str, pattern: list = ['nvn','an','npn','nvn_mod'], spacy_model_name: str = 'en_core_web_lg') -> None:
        """
        This is the pattern finder class which is responsible for extracting grammaticals combinations of 
        nouns as subjects, objects, verbs and prepositions as action and combination of compound nouns and adjectives
        as subjects, objects along with verb as action
        
        args:
        - data (pd.DataFrame): pandas dataframe consisting of textual content
        - textual_col (str): name of column in input pandas dataframe
        - pattern (list, str): patterns to process
        
        return:
        - None
        """
        self._data = data
        self._textual_col = textual_col
        self._pattern = pattern
        self._spacy_model_name = spacy_model_name
        
        self._overall_extract = self._data.copy()
        self._nvn_seg_patterns = None
        self._an_seg_patterns = None
        self._npn_seg_patterns = None
        self._nvn_mod_seg_patterns = None
        
    #region Properties
    @property
    def get_nvn_patterns(self):
        return self._nvn_seg_patterns
    
    @property
    def get_an_patterns(self):
        return self._an_seg_patterns
    
    @property
    def get_npn_patterns(self):
        return self._npn_seg_patterns
    
    @property
    def get_nvn_mod_patterns(self):
        return self._nvn_mod_seg_patterns
    #endregion
    
    def process_patterns(self):
        """
        """
        pass
    
    def extract_seg_nvn(self):
        """
        """
        pass
    
    def extract_seg_an(self):
        """
        """
        pass
    
    def extract_seg_npn(self):
        """
        """
        pass
    
    def extract_seg_nvn_an(self):
        """
        """
        pass