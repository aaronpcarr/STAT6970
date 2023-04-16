import numpy as np
import pandas as pd

data = pd.read_csv("Scripts\BigSix_Tokenized.csv")

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
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(comment_train_feature, train_labels)
ypred = model.predict(comment_test_feature)

from sklearn.metrics import accuracy_score
accuracy_score(ypred, test_labels)
print("Log Reg with Replies:", accuracy_score(ypred, test_labels))

misclassified = np.where(test_labels != ypred)
misclassified

test_labels[5]
ypred[5]
comment_test[5]

coefficients = model.coef_
feature_names = vectorizer.get_feature_names_out()
coef_df = pd.DataFrame(coefficients, columns=feature_names, index=model.classes_)
coef_df.to_csv("model_coefficients.csv")

from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier(class_weight='balanced', max_depth=100)
model2.fit(comment_train_feature, train_labels)
ypred2 = model2.predict(comment_test_feature)

accuracy_score(ypred2, test_labels)
print("Decision Tree with Replies:", accuracy_score(ypred2, test_labels))

misclassified2 = np.where(test_labels != ypred2)
misclassified2

from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(class_weight='balanced', n_estimators = 100, random_state = 1)
model3.fit(comment_train_feature, train_labels)
ypred3 = model3.predict(comment_test_feature)

accuracy_score(ypred3, test_labels)
print("Random Forest with Replies:", accuracy_score(ypred3, test_labels))

misclassified3 = np.where(test_labels != ypred3)
misclassified3

from sklearn.ensemble import GradientBoostingClassifier
model4 = GradientBoostingClassifier(
        max_depth = 4,
        n_estimators = 10,
        learning_rate = 1,
        random_state = 1
)

from sklearn.utils import compute_class_weight
weights = compute_class_weight(class_weight='balanced',classes = np.unique(train_labels),y = train_labels)
sample_weights = np.zeros(len(stem_train))
sample_weights[train_labels == "Arsenal"] = weights[0]
sample_weights[train_labels == "Chelsea"] = weights[1]
sample_weights[train_labels == "Liverpool"] = weights[2]
sample_weights[train_labels == "Manchester City"] = weights[3]
sample_weights[train_labels == "Manchester United"] = weights[4]
sample_weights[train_labels == "Spurs"] = weights[5]
    

model4.fit(comment_train_feature, train_labels,sample_weight= sample_weights)

ypred4 = model4.predict(comment_test_feature)

accuracy_score(ypred4, test_labels)
print("GradientBoosting with Replies:", accuracy_score(ypred4, test_labels))

misclassified4 = np.where(test_labels != ypred4)
misclassified4

import xgboost as xgb
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded_train_labels = label_encoder.fit_transform(train_labels)
encoded_test_labels = label_encoder.transform(test_labels)

class_weights = compute_class_weight(class_weight='balanced',
        classes=np.unique(encoded_train_labels),
        y=encoded_train_labels)
weights = {club : weight for club, weight in zip(np.unique(encoded_train_labels), class_weights)}
model5 = xgb.XGBClassifier(random_state=1)
weights_array = np.array([weights[club] for club in encoded_train_labels])
model5.fit(comment_train_feature, encoded_train_labels, sample_weight=weights_array)
ypred5 = model5.predict(comment_test_feature)

accuracy_score(ypred5, encoded_test_labels)
print("XGB w Replies:", accuracy_score(ypred5, encoded_test_labels))


#Filter out comments that are replies to see 
#if accuracy score will improve
noreply = data[['Comment','stemmed_comments','Club','Reply']]
noreply = noreply.loc[noreply["Reply"] == "No"]
noreply = noreply.rename(columns={'stemmed_comments': 'Stem'})

from sklearn.model_selection import train_test_split
commentr = noreply['Comment'].values
stemr = noreply["Stem"].values
yr = noreply["Club"].values
commentr_train, commentr_test, stemr_train, stemr_test, trainr_labels, testr_labels = train_test_split(commentr, stemr, yr, test_size = 0.20, random_state = 1)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(stemr_train)
vectorizer.vocabulary_

commentr_train_feature = vectorizer.transform(stemr_train)
commentr_test_feature = vectorizer.transform(stemr_test)

from sklearn.linear_model import LogisticRegression
modelr = LogisticRegression(class_weight='balanced', max_iter=1000)
modelr.fit(commentr_train_feature, trainr_labels)
ypredr = modelr.predict(commentr_test_feature)

from sklearn.metrics import accuracy_score
accuracy_score(ypredr, testr_labels)
print("Log Reg w/o Replies:", accuracy_score(ypredr, testr_labels))

from sklearn.tree import DecisionTreeClassifier
modelr2 = DecisionTreeClassifier(class_weight='balanced', max_depth=100)
modelr2.fit(commentr_train_feature, trainr_labels)
ypredr2 = modelr2.predict(commentr_test_feature)

accuracy_score(ypredr2, testr_labels)
print("Decision Tree w/o Replies:", accuracy_score(ypredr2, testr_labels))

from sklearn.ensemble import RandomForestClassifier
modelr3 = RandomForestClassifier(class_weight='balanced', n_estimators = 100, random_state = 1)
modelr3.fit(commentr_train_feature, trainr_labels)
ypredr3 = modelr3.predict(commentr_test_feature)

accuracy_score(ypredr3, testr_labels)
print("RandomForest w/o Replies:", accuracy_score(ypredr3, testr_labels))

from sklearn.ensemble import GradientBoostingClassifier
modelr4 = GradientBoostingClassifier(
        max_depth = 4,
        n_estimators = 10,
        learning_rate = 1,
        random_state = 1
)
weightsr = compute_class_weight(class_weight='balanced',classes = np.unique(trainr_labels),y = trainr_labels)
sample_weightsr = np.zeros(len(stemr_train))
sample_weightsr[trainr_labels == "Arsenal"] = weights[0]
sample_weightsr[trainr_labels == "Chelsea"] = weights[1]
sample_weightsr[trainr_labels == "Liverpool"] = weights[2]
sample_weightsr[trainr_labels == "Manchester City"] = weights[3]
sample_weightsr[trainr_labels == "Manchester United"] = weights[4]
sample_weightsr[trainr_labels == "Spurs"] = weights[5]

modelr4.fit(commentr_train_feature, trainr_labels,sample_weight = sample_weightsr)

ypredr4 = modelr4.predict(commentr_test_feature)

accuracy_score(ypredr4, testr_labels)
print("GradientBoosting w/o Replies:", accuracy_score(ypredr4, testr_labels))

labelr_encoder = LabelEncoder()
encoded_trainr_labels = label_encoder.fit_transform(trainr_labels)
encoded_testr_labels = label_encoder.transform(testr_labels)

class_weightsr = compute_class_weight(class_weight='balanced',
        classes=np.unique(encoded_trainr_labels),
        y=encoded_trainr_labels)
weightsr = {club : weight for club, weight in zip(np.unique(encoded_trainr_labels), class_weightsr)}
modelr5 = xgb.XGBClassifier(random_state=1)
weightsr_array = np.array([weightsr[club] for club in encoded_trainr_labels])
modelr5.fit(commentr_train_feature, encoded_trainr_labels, sample_weight=weightsr_array)
ypredr5 = modelr5.predict(commentr_test_feature)

accuracy_score(ypredr5, encoded_testr_labels)
print("XGBoost w/o Replies:", accuracy_score(ypredr5, encoded_testr_labels))


