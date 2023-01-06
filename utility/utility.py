__author__ = "konwar.m"
__copyright__ = "Copyright 2023, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import spacy

def load_spacy_model(model_name, exclude_list=None):
    """
    This method loads spacy model based on language name else download it and
    load it
    model_name (str): model name to load
    exclude_list (list, str): 
    """
    excluded_steps = ["tagger", "parser", "ner", "entity_linker", 
                      "entity_ruler", "textcat", "morphologizer",
                      "attribute_ruler", "senter", "sentencizer", 
                      "token2vec", "transformer"] if exclude_list is None else exclude_list
    try:
        spacy_model = spacy.load(model_name, exclude=excluded_steps)
    except OSError:
        spacy.cli.download(model_name)
        spacy_model = spacy.load(model_name, exclude=excluded_steps)
    finally:
        return spacy_model