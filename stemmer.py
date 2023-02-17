import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re

def tokenize(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    return tokens

stemmer = SnowballStemmer('english')

def stem(tokens):
    stems = [stemmer.stem(token) for token in tokens]
    return stems

# Having trouble with some of the characters, for now I'm having it ignore the errors but need to look into this later.

comments = pd.read_csv('\STAT6970\BigSix.csv')
#Remove removed or deleted comments
comments = comments[~comments['Comment'].isin(['Deleted', 'Removed'])]
comments['tokenized_comments'] = comments['Comment'].apply(tokenize)
comments['stemmed_comments'] = comments['tokenized_comments'].apply(stem)
comments.to_csv("\STAT6970\BigSix_Tokenized.csv")

print(comments['stemmed_comments'])