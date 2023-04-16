#Run the Model Training.py file before using this script.

arsenal_test = vectorizer.transform(stem_test[np.where(test_labels == "Arsenal")])
chelsea_test = vectorizer.transform(stem_test[np.where(test_labels == "Chelsea")])
liverpool_test = vectorizer.transform(stem_test[np.where(test_labels == "Liverpool")])
mancity_test = vectorizer.transform(stem_test[np.where(test_labels == "Manchester City")])
manutd_test = vectorizer.transform(stem_test[np.where(test_labels == "Manchester United")])
spurs_test = vectorizer.transform(stem_test[np.where(test_labels == "Spurs")])

#Accuracy Score by Club for Logistic Regression Model
arsenal_pred = model.predict(arsenal_test)
chelsea_pred = model.predict(chelsea_test)
liverpool_pred = model.predict(liverpool_test)
mancity_pred = model.predict(mancity_test)
manutd_pred = model.predict(manutd_test)
spurs_pred = model.predict(spurs_test)

accuracy_score(arsenal_pred, test_labels[np.where(test_labels == "Arsenal")])
accuracy_score(chelsea_pred, test_labels[np.where(test_labels == "Chelsea")])
accuracy_score(liverpool_pred, test_labels[np.where(test_labels == "Liverpool")])
accuracy_score(mancity_pred, test_labels[np.where(test_labels == "Manchester City")])
accuracy_score(manutd_pred, test_labels[np.where(test_labels == "Manchester United")])
accuracy_score(spurs_pred, test_labels[np.where(test_labels == "Spurs")])

#Accuracy Score by Club for Random Forest Model
arsenal_pred3 = model3.predict(arsenal_test)
chelsea_pred3 = model3.predict(chelsea_test)
liverpool_pred3 = model3.predict(liverpool_test)
mancity_pred3 = model3.predict(mancity_test)
manutd_pred3 = model3.predict(manutd_test)
spurs_pred3 = model3.predict(spurs_test)

accuracy_score(arsenal_pred3, test_labels[np.where(test_labels == "Arsenal")])
accuracy_score(chelsea_pred3, test_labels[np.where(test_labels == "Chelsea")])
accuracy_score(liverpool_pred3, test_labels[np.where(test_labels == "Liverpool")])
accuracy_score(mancity_pred3, test_labels[np.where(test_labels == "Manchester City")])
accuracy_score(manutd_pred3, test_labels[np.where(test_labels == "Manchester United")])
accuracy_score(spurs_pred3, test_labels[np.where(test_labels == "Spurs")])

#Accuracy Score by Club for Gradient Boost Model
arsenal_pred4 = model4.predict(arsenal_test)
chelsea_pred4 = model4.predict(chelsea_test)
liverpool_pred4 = model4.predict(liverpool_test)
mancity_pred4 = model4.predict(mancity_test)
manutd_pred4 = model4.predict(manutd_test)
spurs_pred4 = model4.predict(spurs_test)

accuracy_score(arsenal_pred4, test_labels[np.where(test_labels == "Arsenal")])
accuracy_score(chelsea_pred4, test_labels[np.where(test_labels == "Chelsea")])
accuracy_score(liverpool_pred4, test_labels[np.where(test_labels == "Liverpool")])
accuracy_score(mancity_pred4, test_labels[np.where(test_labels == "Manchester City")])
accuracy_score(manutd_pred4, test_labels[np.where(test_labels == "Manchester United")])
accuracy_score(spurs_pred4, test_labels[np.where(test_labels == "Spurs")])

#Accuracy Score by Club for XGBoost Model
arsenal_test5 = vectorizer.transform(stem_test[np.where(encoded_test_labels == 0)])
chelsea_test5 = vectorizer.transform(stem_test[np.where(encoded_test_labels == 1)])
liverpool_test5 = vectorizer.transform(stem_test[np.where(encoded_test_labels == 2)])
mancity_test5 = vectorizer.transform(stem_test[np.where(encoded_test_labels == 3)])
manutd_test5 = vectorizer.transform(stem_test[np.where(encoded_test_labels == 4)])
spurs_test5 = vectorizer.transform(stem_test[np.where(encoded_test_labels == 5)])

arsenal_pred5 = model5.predict(arsenal_test5)
chelsea_pred5 = model5.predict(chelsea_test5)
liverpool_pred5 = model5.predict(liverpool_test5)
mancity_pred5 = model5.predict(mancity_test5)
manutd_pred5 = model5.predict(manutd_test5)
spurs_pred5 = model5.predict(spurs_test5)

accuracy_score(arsenal_pred5, encoded_test_labels[np.where(encoded_test_labels == 0)])
accuracy_score(chelsea_pred5,  encoded_test_labels[np.where(encoded_test_labels == 1)])
accuracy_score(liverpool_pred5,  encoded_test_labels[np.where(encoded_test_labels == 2)])
accuracy_score(mancity_pred5,  encoded_test_labels[np.where(encoded_test_labels == 3)])
accuracy_score(manutd_pred5,  encoded_test_labels[np.where(encoded_test_labels == 4)])
accuracy_score(spurs_pred5,  encoded_test_labels[np.where(encoded_test_labels == 5)])

march = pd.read_csv("BigSixMarch_Tokenized.csv")
april = pd.read_csv("BigSixApril_Tokenized.csv")

march_test = vectorizer.transform(march["stemmed_comments"].values)
april_test = vectorizer.transform(april["stemmed_comments"].values)

march_pred = model.predict(march_test)
april_pred = model.predict(april_test)

accuracy_score(march_pred,march["Club"])
accuracy_score(april_pred,april["Club"])

march_pred3 = model3.predict(march_test)
april_pred3 = model3.predict(april_test)

accuracy_score(march_pred3,march["Club"])
accuracy_score(april_pred3,april["Club"])

march_pred4 = model4.predict(march_test)
april_pred4 = model4.predict(april_test)

accuracy_score(march_pred4,march["Club"])
accuracy_score(april_pred4,april["Club"])

label_encoder.fit_transform(

march_pred5 = model5.predict(march_test)
april_pred5 = model5.predict(april_test)

accuracy_score(march_pred5,label_encoder.fit_transform(march["Club"]))
accuracy_score(april_pred5,label_encoder.fit_transform(april["Club"]))
