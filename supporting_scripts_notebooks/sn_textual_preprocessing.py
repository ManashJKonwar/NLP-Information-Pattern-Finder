__author__ = "konwar.m"
__copyright__ = "Copyright 2023, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

"""
This script is intended to perform textual preprocessing on intended texts. It performs the following operations
- Remove mentions and hashtags  
- Remove URLs  
- Remove contractions  
- Remove stopwords and punctuations  
- Lemmatize all words amd lower case each of them  
- Remove redundant domain specific words  
- Remove extra spaces 
"""

contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he shall",
    "he'll've": "he shall have",
    "he's": "he has",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has",
    "i'd": "I had",
    "i'd've": "I would have",
    "i'll": "I shall",
    "i'll've": "I shall have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it shall",
    "it'll've": "it shall have",
    "it's": "it has",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she shall",
    "she'll've": "she shall have",
    "she's": "she has",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that has",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they had",
    "they'd've": "they would have",
    "they'll": "they shall",
    "they'll've": "they shall have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall",
    "what'll've": "what shall have",
    "what're": "what are",
    "what's": "what has",
    "what've": "what have",
    "when's": "when has",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has",
    "where've": "where have",
    "who'll": "who shall",
    "who'll've": "who shall have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you hadd",
    "you'd've": "you would have",
    "you'll": "you shall",
    "you'll've": "you shall have",
    "you're": "you are",
    "you've": "you have",
    "ve": "have",
}

porter = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

def convert_lower(text):
    result = text.lower()
    return result

def remove_extra_spaces(text):
    result = re.sub(' +', ' ', text).strip()
    return result

def remove_urls(text):
    result = re.sub('http://\S+|https://\S+', '', text)
    return result

def remove_mentions_hashtags(text):
    # removing mentions
    result = re.sub("@[A-Za-z0-9_]+","", text)
    # removing hastags
    result = re.sub("#[A-Za-z0-9_]+","", result)
    return result

def remove_contractions(text):
    # removing contractions
    for word in text.split():
        if word.lower() in contractions:
            text = text.replace(word, contractions[word.lower()])
    return text

def remove_stopwords_punc_nos(text, 
                              remove_stopwords_flag=False, 
                              punc_2_remove=string.punctuation, 
                              remove_digits_flag=True, 
                              remove_pattern_punc_flag=False):
    sentence_list = sent_tokenize(text)
    
    regex_pattern = r"(?<!\d)[.,;:](?!\d)"
    
    # removing stopwords if flag is set as True
    if remove_stopwords_flag:
        result_list = [
            remove_stopwords(sentence)\
            .translate(str.maketrans('','', punc_2_remove))\
            .strip() for sentence in sentence_list
        ]
    else:
        result_list = [
            sentence.translate(str.maketrans('','', punc_2_remove))\
            .strip() for sentence in sentence_list
        ]
        
    # removing all digits if flag is set as True
    if remove_digits_flag:
        result_list = [
            sentence.translate(str.maketrans('','', string.digits))\
            .strip() for sentence in result_list
        ]
        
    if remove_pattern_punc_flag:
        result_list = [
        re.sub(regex_pattern, '', sentence, 0) for sentence in result_list
    ]
        
    result = '. '.join(result_list)
    return result
    
def stem_lower(text):
    # stemming and lower casing all words
    token_words = word_tokenize(text)
    stem_sentence = [porter.stem(word) for word in token_words]
    return ' '.join(stem_sentence)

def lemma_lower(text):
    # lower casing all words first
    text = text.lower()
    # lemmatizing the text
    token_words = word_tokenize(text)
    lemma_sentence = [wordnet_lemmatizer.lemmatize(word) for word in token_words]
    lemma_sentence = ' '.join(lemma_sentence)
    # remove spaces before punctuation (full stop) of each sentence
    result = re.sub(r'\s([?.!"](?:\s|$))', r'\1', lemma_sentence)
    return result

def spell_checker(text):
    # Spell checker function
    def check_word_spelling(word):
        word = Word(word)
        result = word.spellcheck()
        if word == result[0][0]:
            # print(f'Spelling of "{word}" is correct!')
            return word
        else:
            if domain_flag:
                if word in domain_acceptable_words:
                    return word
                elif word in domain_removable_words:
                    # print(True)
                    # print(word)
                    return ''
            # print(f'Spelling of "{word}" is not correct!')
            # print(f'Correct spelling of "{word}": "{result[0][0]}" (with {result[0][1]} confidence).')
            if result[0][1]>0.9:
                return result[0][0]
            else:
                return word
    
    # Call spell checker for each word from the sentence
    words = text.split()
    final_text = []
    for word in words:
        final_text.append(check_word_spelling(word))
    return ' '.join(final_text)