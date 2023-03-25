import numpy as np
import pandas as pd

data = pd.read_csv("BigSix_Tokenized.csv")

bigsix = data[['Comment','stemmed_comments','Club']]

bigsix = bigsix.rename(columns={'stemmed_comments': 'Stem'})


#Split data in to train and test. Train on stemmed comments
#Raw text comments included into split so that interesting comments
#can be read easier.
from sklearn.model_selection import train_test_split
comment = bigsix['Comment'].values
stem = bigsix["Stem"].values
y = bigsix["Club"].values
comment_train, comment_test, stem_train, stem_test, train_labels, test_labels = train_test_split(comment, stem, y, test_size = 0.20, random_state = 1)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(stem_train)
vectorizer.vocabulary_

comment_train_feature = vectorizer.transform(stem_train)
comment_test_feature = vectorizer.transform(stem_test)
comment_train_feature.shape
comment_test_feature.shape

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(comment_train_feature, train_labels)
ypred = model.predict(comment_test_feature)

from sklearn.metrics import accuracy_score
accuracy_score(ypred, test_labels)

misclassified = np.where(test_labels != ypred)
misclassified

test_labels[5]
ypred[5]
comment_test[5]

from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier(max_depth=10)
model2.fit(comment_train_feature, train_labels)
ypred2 = model2.predict(comment_test_feature)

accuracy_score(ypred2, test_labels)

misclassified2 = np.where(test_labels != ypred2)
misclassified2

from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators = 100, random_state = 1)
model3.fit(comment_train_feature, train_labels)
ypred3 = model2.predict(comment_test_feature)

accuracy_score(ypred3, test_labels)

misclassified3 = np.where(test_labels != ypred3)
misclassified3

from sklearn.ensemble import GradientBoostingClassifier
model4 = GradientBoostingClassifier(
        max_depth = 4,
        n_estimators = 10,
        learning_rate = 1,
        random_state = 1
)

model4.fit(comment_train_feature, train_labels)

ypred4 = model4.predict(comment_test_feature)

accuracy_score(ypred4, test_labels)

misclassified4 = np.where(test_labels != ypred4)
misclassified4