import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

comments = pd.read_csv("\STAT6970\BigSix_Tokenized.csv")

X_train, X_test, y_train, y_test = train_test_split(comments['tokenized_comments'], comments['Club'], test_size=0.2, random_state=1)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train.apply(' '.join))
X_test = vectorizer.fit_trainsform(X_test.apply(' '.join))

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print('Accuracy:', accuracy)
print('F1 Score:', f1)


print(X_train)