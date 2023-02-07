import pandas as pd
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# Having trouble with some of the characters, for now I'm having it ignore the errors but need to look into this later.

comments = pd.read_csv('\STAT6970\BigSix.csv')
comments['Stemmed Comment'] = comments['Comment'].apply(lambda x: stemmer.stem(x))
comments['Tokenized Comment'] = comments['Stemmed Comment'].apply(lambda x: nltk.word_tokenize(x))
comments.to_csv("\STAT6970\BigSix_Tokenized.csv")
