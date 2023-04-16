#RUn the Model Training.py file before using this script.

arsenal_test = vectorizer.transform(stem_test[np.where(test_labels == "Arsenal")])
chelsea_test = vectorizer.transform(stem_test[np.where(test_labels == "Chelsea")])
liverpool_test = vectorizer.transform(stem_test[np.where(test_labels == "Liverpool")])
mancity_test = vectorizer.transform(stem_test[np.where(test_labels == "Manchester City")])
manutd_test = vectorizer.transform(stem_test[np.where(test_labels == "Manchester United")])
spurs_test = vectorizer.transform(stem_test[np.where(test_labels == "Spurs")])

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
