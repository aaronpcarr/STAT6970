import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

comments = pd.read_csv("D:\STAT6970\Datasets\BigSix.csv")

def distinguish(club):
    target_club = comments[comments['Club'] == club]
    other_clubs = comments[comments['Club'] != club]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(target_club["Comment"])
    target_tfidf = vectorizer.transform(target_club['Comment'])
    vectorizer.fit(other_clubs["Comment"])
    other_tfidf = vectorizer.transform(other_clubs['Comment'])
    target_mean_tfidf = target_tfidf.mean(axis=0)
    other_mean_tfidf = other_tfidf.mean(axis=0)
    diff_tfidf = target_mean_tfidf - other_mean_tfidf
    words = vectorizer.get_feature_names()
    sorted_words = sorted(zip(words, diff_tfidf.tolist()[0]), key=lambda x: x[1], reverse=True)
    N = 10
    top_words = []
    for word, score in sorted_words[:N]:
        top_words.append((word, score))
    return pd.DataFrame(top_words, columns=['Word', 'Score'])

results = comments.groupby('Club').apply(distinguish)
print(results)

