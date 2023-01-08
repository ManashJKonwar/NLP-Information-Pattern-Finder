__author__ = "konwar.m"
__copyright__ = "Copyright 2023, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os
import sys
import pandas as pd
from tqdm import tqdm
file_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(file_dir)

from supporting_scripts_notebooks.sn_textual_preprocessing import *
from utility.utility import load_spacy_model

# Function for rule 1: noun(subject), verb, noun(object)
def rule_nvn(text: str, spacy_loaded_model=None) -> list:
    """
    This function is responsible for extracting all possible combinations of 
    NOUN / PROPER NOUN / PRONOUN (subject) <-> VERB <-> NOUN / PROPER NOUN that is seen in the 
    input text.

    args:
        text (str): Input text for which NVN phrases needs to be extracted
        spacy_loaded_model (spacy model object, optional): spacy model to be used for accessing the POS tags. Defaults to None.

    returns:
        list: list of dictionaries where each dictionary is representation of one NVN phrase being detected
        e.g. [{'phrase': 'ONS revise rate', 'verb': 'revise'},
            {'phrase': 'number report figures', 'verb': 'report'},
            {'phrase': 'retailers endure Christmas', 'verb': 'endure'}]
    """
    doc = spacy_loaded_model(text)
    sent = []
    for token in doc:
        # if the token is a verb
        if (token.pos_=='VERB'):
            phrase =''
            
            # only extract noun or pronoun subjects
            for sub_tok in token.lefts:
                if (sub_tok.dep_ in ['nsubj','nsubjpass']) and (sub_tok.pos_ in ['NOUN','PROPN','PRON']):
                    # add subject to the phrase
                    phrase += sub_tok.text
                    # save the root of the verb in phrase
                    phrase += ' '+token.lemma_ 
                    # check for noun or pronoun direct objects
                    for sub_tok in token.rights:
                        # save the object in the phrase
                        if (sub_tok.dep_ in ['dobj']) and (sub_tok.pos_ in ['NOUN','PROPN']):
                            phrase += ' '+sub_tok.text
                            sent.append({'phrase': phrase, 'verb': token.lemma_})
    return sent

# Function for rule 2: adjective noun
def rule_an(text: str, spacy_loaded_model=None) -> list:
    """
    This function is responsible for extracting all possible combinations of 
    ADJECTIVE / COMPOUND <-> NOUN (subject, object, nominal subject, passive nominal subject)

    args:
        text (str): Input text for which AN phrases needs to be extracted
        spacy_loaded_model (spacy model object, optional): spacy model to be used for accessing the POS tags. Defaults to None.

    returns:
        list : list of dictionaries where each dictionary is representation of one AN phrase being detected
        e.g. [{'phrase': 'significant growth', 'noun': 'growth'},
            {'phrase': 'earlier caution', 'noun': 'caution'},
            {'phrase': 'poor December figures', 'noun': 'figures'}]
    """
    doc = spacy_loaded_model(text)
    pat = []
    
    # iterate over tokens
    for token in doc:
        phrase = ''
        # if the word is a subject noun or an object noun
        if (token.pos_ == 'NOUN')\
            and (token.dep_ in ['dobj','pobj','nsubj','nsubjpass']):
            
            # iterate over the children nodes
            for subtoken in token.children:
                # if word is an adjective or has a compound dependency
                if (subtoken.pos_ == 'ADJ') or (subtoken.dep_ == 'compound'):
                    phrase += subtoken.text + ' '
                    
            if len(phrase)!=0:
                phrase += token.text
             
        if  len(phrase)!=0:
            pat.append({'phrase':phrase, 'noun': token.text})
    return pat

# Function for rule 3: noun, preposition, noun
def rule_npn(text: str, spacy_loaded_model=None) -> list:
    """
    This function is responsible for extracting all possible combinations of 
    NOUN <-> PREPOSITION <-> NOUN / PROPER NOUN

    args:
        text (str): Input text for which NPN phrases needs to be extracted
        spacy_loaded_model (_type_, optional): spacy model to be used for accessing the POS tags. Defaults to None.

    returns:
        list: list of dictionaries where each dictionary is representation of one NPN phrase being detected
        e.g. [{'phrase': 'number of retailers', 'preposition': 'of'},
            {'phrase': 'caution from King', 'preposition': 'from'},
            {'phrase': 'way below booms', 'preposition': 'below'}]
    """
    doc = spacy_loaded_model(text)
    sent = []
    
    for token in doc:
        # look for prepositions
        if token.pos_=='ADP':
            phrase = ''
            # if its head word is a noun
            if token.head.pos_=='NOUN':
                # append noun and preposition to phrase
                phrase += token.head.text
                phrase += ' '+token.text

                # check the nodes to the right of the preposition
                for right_tok in token.rights:
                    # append if it is a noun or proper noun
                    if (right_tok.pos_ in ['NOUN','PROPN']):
                        phrase += ' '+right_tok.text
                
                if len(phrase)>2:
                    sent.append({'phrase':phrase, 'preposition': token.text})
    return sent

# Function for rule 4: Combination of rulw 1 + rule 2
def rule_ad_mod(doc, text: str, index: int) -> str:
    # doc = nlp_spacy_en_model(text)
    phrase = ''
    
    for token in doc:
        if token.i == index:
            for subtoken in token.children:
                if (subtoken.pos_ == 'ADJ'):
                    phrase += ' '+subtoken.text
            break
    return phrase

def rule_nvn_mod(text: str, spacy_loaded_model=None) -> list:
    """
    This function is responsible for extracting all possible combinations of 
    COMPOUND / ADJ <-> NOUN / PROPER NOUN / PRONOUN (subject) <-> VERB <-> COMPOUND / ADJ <-> NOUN / PROPER NOUN that is seen in the 
    input text.

    args:
        text (str): Input text for which compound / adjective NVN phrases needs to be extracted
        spacy_loaded_model (spacy model object, optional): spacy model to be used for accessing the POS tags. Defaults to None.

    returns:
        list: list of dictionaries where each dictionary is representation of one NVN phrase being detected
        e.g. [{'phrase': ' ONS revise annual rate', 'verb': 'revise'},
            {'phrase': ' number report poor figures', 'verb': 'report'},
            {'phrase': ' retailers endure tougher Christmas', 'verb': 'endure'}]
    """
    doc = spacy_loaded_model(text)
    sent = []
    
    for token in doc:
        # root word
        if (token.pos_=='VERB'):
            phrase =''
            
            # only extract noun or pronoun subjects
            for sub_tok in token.lefts:
                if (sub_tok.dep_ in ['nsubj','nsubjpass']) and (sub_tok.pos_ in ['NOUN','PROPN','PRON']):
                    adj = rule_ad_mod(doc, text, sub_tok.i)
                    phrase += adj + ' ' + sub_tok.text

                    # save the root word of the word
                    phrase += ' '+token.lemma_ 

                    # check for noun or pronoun direct objects
                    for sub_tok in token.rights:
                        if (sub_tok.dep_ in ['dobj']) and (sub_tok.pos_ in ['NOUN','PROPN']):
                            adj = rule_ad_mod(doc, text, sub_tok.i)
                            # add adj based noun
                            phrase += adj+' '+sub_tok.text
                            sent.append({'phrase':phrase, 'verb':token.lemma_})
    return sent

class PatternFinder:
    def __init__(self, data: pd.DataFrame, textual_col: str, pattern_collection: list = ['nvn','an','npn','nvn_mod'], spacy_model_name: str = 'en_core_web_lg') -> None:
        """
        This is the pattern finder class which is responsible for extracting grammaticals combinations of 
        nouns as subjects, objects, verbs and prepositions as action and combination of compound nouns and adjectives
        as subjects, objects along with verb as action
        
        args:
        - data (pd.DataFrame): pandas dataframe consisting of textual content
        - textual_col (str): name of column in input pandas dataframe
        - pattern_collection (list, str): patterns to process
        
        return:
        - None
        """
        self._data = data
        self._textual_col = textual_col
        self._pattern_collection = pattern_collection
        self._spacy_model_name = spacy_model_name
        self._spacy_loaded_model = load_spacy_model(self._spacy_model_name, exclude_list=[])
        
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
        This method is to run processes which would extract the input patterns
        decided, merge them with thr original dataframe and also store them as output
        extract
        
        args:
        - None
        
        returns:
        - None
        """
        if 'nvn' in self._pattern_collection:
            tqdm.pandas(desc='Extracting NVN phrases')
            self._overall_extract['NVN_PHRASES'] = self._overall_extract[self._textual_col].progress_apply(lambda x : rule_nvn(x, spacy_loaded_model = self._spacy_loaded_model))
            
        if 'an' in self._pattern_collection:
            tqdm.pandas(desc='Extracting AN phrases')
            self._overall_extract['AN_PHRASES'] = self._overall_extract[self._textual_col].progress_apply(lambda x : rule_an(x, spacy_loaded_model = self._spacy_loaded_model))
        
        if 'npn' in self._pattern_collection:
            tqdm.pandas(desc='Extracting NPN phrases')
            self._overall_extract['NPN_PHRASES'] = self._overall_extract[self._textual_col].progress_apply(lambda x : rule_npn(x, spacy_loaded_model = self._spacy_loaded_model))
    
        if 'nvn_mod' in self._pattern_collection:
            tqdm.pandas(desc='Extracting NVN with Adjectives / Compound nouns based phrases')
            self._overall_extract['NVN_MOD_PHRASES'] = self._overall_extract[self._textual_col].progress_apply(lambda x : rule_nvn_mod(x, spacy_loaded_model = self._spacy_loaded_model))
    
    def extract_seg_nvn(self):
        """
        This method is to segregate the extracted NVN phrases into 3 separate columns noun1, verb and noun2.
        
        args:
        - None
        
        return:
        - None
        """
        sample_data_copy = self._overall_extract[['ARTICLES','CATEGORIES','PREPROCESSED_TEXT','NVN_PHRASES']].copy().reset_index(drop=True)
        # selecting non-empty output rows
        nvn_sample_data = pd.DataFrame(columns=sample_data_copy.columns)
        try:
            for row in tqdm(range(len(sample_data_copy)), desc='Selecting non empty rows'):
                if len(sample_data_copy.loc[row,'NVN_PHRASES'])!=0:
                    nvn_sample_data = pd.concat([nvn_sample_data, pd.DataFrame([sample_data_copy.loc[row,:]])], ignore_index=True)

            # reset the index
            nvn_sample_data.reset_index(inplace=True)
            nvn_sample_data.drop('index', axis=1, inplace=True)   
            print(nvn_sample_data.shape)
        except Exception as ex:
            print(ex)
            pass
        
        verb_dict = dict()
        dis_dict = dict()
        dis_list = []
        try:
            # iterating over all the sentences
            for i in range(len(nvn_sample_data)):
                
                # sentence containing the output
                sentence = nvn_sample_data.loc[i,'PREPROCESSED_TEXT']
                # catgeory info
                category = nvn_sample_data.loc[i,'CATEGORIES']
                # output of the sentence
                output = nvn_sample_data.loc[i,'NVN_PHRASES']
                
                # iterating over all the outputs from the sentence
                for sent in output:
                    # separate subject, verb and object
                    n1, v, n2 = sent['phrase'].split(sent['verb'])[0], sent['verb'], sent['phrase'].split(sent['verb'])[1]
                    
                    # append to list, along with the sentence
                    dis_dict = {
                        'PREPROCESSED_TEXT':sentence,
                        'CATEGORY':category,
                        'NOUN1':n1,
                        'VERB':v,
                        'NOUN2':n2}
                    dis_list.append(dis_dict)
                    
                    # counting the number of sentences containing the verb
                    verb = sent['phrase'].split()[1]
                    if verb in verb_dict:
                        verb_dict[verb]+=1
                    else:
                        verb_dict[verb]=1
            
            df_nvn_sep = pd.DataFrame(dis_list)
        except Exception as ex:
            print(ex)
        finally:
            self._nvn_seg_patterns = df_nvn_sep.copy()
        
    def extract_seg_an(self):
        """
        This method is to segregate the extracted AN phrases into 2 separate columns adjective and noun.
        
        args:
        - None
        
        return:
        - None
        """
        # selecting non-empty output rows
        sample_data_copy = self._overall_extract[['ARTICLES','CATEGORIES','PREPROCESSED_TEXT','AN_PHRASES']].copy().reset_index(drop=True)
        an_sample_data = pd.DataFrame(columns=sample_data_copy.columns)

        try:
            for row in tqdm(range(len(sample_data_copy)), desc='Selecting non empty rows'):
                if len(sample_data_copy.loc[row,'AN_PHRASES'])!=0:
                    an_sample_data = pd.concat([an_sample_data, pd.DataFrame([sample_data_copy.loc[row,:]])], ignore_index=True)

            # reset the index
            an_sample_data.reset_index(inplace=True)
            an_sample_data.drop('index', axis=1, inplace=True)   
            print(an_sample_data.shape)
        except Exception as ex:
            print(ex)
            pass
        
        noun_dict = dict()
        dis_dict = dict()
        dis_list = []
        try:
            # iterating over all the sentences
            for i in range(len(an_sample_data)):
                
                # sentence containing the output
                sentence = an_sample_data.loc[i,'PREPROCESSED_TEXT']
                # catgeory info
                category = an_sample_data.loc[i,'CATEGORIES']
                # output of the sentence
                output = an_sample_data.loc[i,'AN_PHRASES']
                
                # iterating over all the outputs from the sentence
                for sent in output:
                    # separate adjective and noun
                    adj, n = ''.join([item.strip() for item in sent['phrase'].split(sent['noun'])]), sent['noun']
                    
                    # append to list, along with the sentence
                    dis_dict = {
                        'PREPROCESSED_TEXT':sentence,
                        'CATEGORY':category,
                        'ADJ':adj,
                        'NOUN':n}
                    dis_list.append(dis_dict)
                    
                    # counting the number of sentences containing the noun
                    noun = sent['noun']
                    if noun in noun_dict:
                        noun_dict[noun]+=1
                    else:
                        noun_dict[noun]=1

            df_an_sep = pd.DataFrame(dis_list)
        except Exception as ex:
            print(ex)
        finally:
            self._an_seg_patterns = df_an_sep.copy()
            
    def extract_seg_npn(self):
        """
        This method is to segregate the extracted AN phrases into 2 separate columns adjective and noun.
        
        args:
        - None
        
        return:
        - None
        """
        # selecting non-empty output rows
        sample_data_copy = self._overall_extract[['ARTICLES','CATEGORIES','PREPROCESSED_TEXT','NPN_PHRASES']].copy().reset_index(drop=True)
        npn_sample_data = pd.DataFrame(columns=sample_data_copy.columns)

        try:
            for row in tqdm(range(len(sample_data_copy)), desc='Selecting non empty rows'):
                if len(sample_data_copy.loc[row,'NPN_PHRASES'])!=0:
                    npn_sample_data = pd.concat([npn_sample_data, pd.DataFrame([sample_data_copy.loc[row,:]])], ignore_index=True)

            # reset the index
            npn_sample_data.reset_index(inplace=True)
            npn_sample_data.drop('index', axis=1, inplace=True)   
            print(npn_sample_data.shape)
        except Exception as ex:
            print(ex)
            pass
        
        preposition_dict = dict()
        dis_dict = dict()
        dis_list = []

        try:
            # iterating over all the sentences
            for i in range(len(npn_sample_data)):
                
                # sentence containing the output
                sentence = npn_sample_data.loc[i,'PREPROCESSED_TEXT']
                # catgeory info
                category = npn_sample_data.loc[i,'CATEGORIES']
                # output of the sentence
                output = npn_sample_data.loc[i,'NPN_PHRASES']
                
                # iterating over all the outputs from the sentence
                for sent in output:
                    # separate subject, verb and object
                    n1, prep, n2 = sent['phrase'].split()[:1], sent['phrase'].split()[1], sent['phrase'].split()[2:]
                    
                    # append to list, along with the sentence
                    dis_dict = {
                        'PREPROCESSED_TEXT':sentence,
                        'CATEGORY':category,
                        'NOUN1':n1,
                        'PREPOSITION':prep,
                        'NOUN2':n2}
                    dis_list.append(dis_dict)
                    
                    # counting the number of sentences containing the verb
                    preposition = sent['phrase'].split()[1]
                    if prep in preposition_dict:
                        preposition_dict[prep]+=1
                    else:
                        preposition_dict[prep]=1

            df_npn_sep = pd.DataFrame(dis_list)
        except Exception as ex:
            print(ex)
        finally:
            self._npn_seg_patterns = df_npn_sep.copy()  
                    
    def extract_seg_nvn_an(self):
        """
        """
        pass
    
if __name__ == "__main__":
    # Test file path
    test_data_path = os.path.join('input','news_articles_dataset.csv')
    
    # File reading configurations
    sample_frac = 0.1
    spacy_model_name = 'en_core_web_lg'
    
    # Reading input file
    input_data = pd.read_csv(test_data_path)
    input_data.columns = [col_name.upper() for col_name in input_data.columns]
    print(input_data.shape)
    
    # Performing stratified sampling for the input data
    sample_data = input_data.groupby('CATEGORIES', group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))
    print(sample_data.shape)
    
    # Preprocessing data before finding patterns
    def preprocess_text(text):
        result = remove_urls(text)
        result = remove_mentions_hashtags(result)
        result = remove_contractions(result)
        result = remove_stopwords_punc_nos(result, 
                                        remove_stopwords_flag=False, 
                                        punc_2_remove=string.punctuation.replace('-','').replace('%','').replace('.',''), 
                                        remove_digits_flag=False,
                                        remove_pattern_punc_flag=True)
        result = remove_extra_spaces(result)
        return result
    
    tqdm.pandas(desc='Preprocessing Raw Texts')
    sample_data['PREPROCESSED_TEXT'] = sample_data.ARTICLES.progress_apply(lambda x: preprocess_text(x))
    
    # Implementing Pattern finder class
    pattern_finder_instance = PatternFinder(data=sample_data, textual_col='PREPROCESSED_TEXT')
    pattern_finder_instance.process_patterns() 
    
    # Implement Segregating of NVN Phrases
    pattern_finder_instance.extract_seg_nvn()
    
    # Implement Segregating of AN Phrases
    pattern_finder_instance.extract_seg_an()
    
    # Implement Segregating of NPN Phrases
    pattern_finder_instance.extract_seg_npn()