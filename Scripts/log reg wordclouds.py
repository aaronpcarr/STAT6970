import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

csv_file = "model_coefficients.csv"
data = pd.read_csv(csv_file, index_col=0)

class_index = 0

feature_names = data.columns.values
coefficients = data.iloc[class_index].values

positive_coefficients = {k: v for k, v in zip(feature_names, coefficients) if v > 0}
positive_normalized_coefficients = {k: v / np.sum(list(positive_coefficients.values())) for k, v in positive_coefficients.items()}

positive_wordcloud = WordCloud(width=800, height = 800, background_color="white", colormap="coolwarm")
positive_wordcloud.generate_from_frequencies(positive_normalized_coefficients)

plt.figure(figsize=(10, 10))
plt.imshow(positive_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

negative_coefficients = {k: v for k, v in zip(feature_names, coefficients) if v < 0}
negative_normalized_coefficients = {k: v / np.sum(list(negative_coefficients.values())) for k, v in negative_coefficients.items()}

negative_wordcloud = WordCloud(width=800, height = 800, background_color="white", colormap="coolwarm")
negative_wordcloud.generate_from_frequencies(negative_normalized_coefficients)

plt.figure(figsize=(10, 10))
plt.imshow(negative_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()