import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

csv_file = "model_coefficients.csv"
data = pd.read_csv(csv_file, index_col=0)

#Arsenal
class_index = 0

feature_names = data.columns.values
coefficients = data.iloc[class_index].values

positive_coefficients = {k: v for k, v in zip(feature_names, coefficients) if v > 0}
positive_normalized_coefficients = {k: v / np.sum(list(positive_coefficients.values())) for k, v in positive_coefficients.items()}

positive_wordcloud = WordCloud(width = 950, height = 950,
                background_color ='red',
                colormap= 'Greys')

positive_wordcloud.generate_from_frequencies(positive_normalized_coefficients)

plt.figure(figsize=(10, 10))
plt.imshow(positive_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#Chelsea
class_index = 1

feature_names = data.columns.values
coefficients = data.iloc[class_index].values

positive_coefficients = {k: v for k, v in zip(feature_names, coefficients) if v > 0}
positive_normalized_coefficients = {k: v / np.sum(list(positive_coefficients.values())) for k, v in positive_coefficients.items()}

positive_wordcloud = WordCloud(width = 950, height = 950,
                background_color ='blue',
                colormap= 'Greys')
positive_wordcloud.generate_from_frequencies(positive_normalized_coefficients)

plt.figure(figsize=(10, 10))
plt.imshow(positive_wordcloud, interpolation="bilinear")

#Liverpool
class_index = 2

feature_names = data.columns.values
coefficients = data.iloc[class_index].values

positive_coefficients = {k: v for k, v in zip(feature_names, coefficients) if v > 0}
positive_normalized_coefficients = {k: v / np.sum(list(positive_coefficients.values())) for k, v in positive_coefficients.items()}

positive_wordcloud = WordCloud(width = 950, height = 950,
                background_color ='maroon',
                colormap= 'BuGn_r')
positive_wordcloud.generate_from_frequencies(positive_normalized_coefficients)

plt.figure(figsize=(10, 10))
plt.imshow(positive_wordcloud, interpolation="bilinear")

#Manchester City
class_index = 3

feature_names = data.columns.values
coefficients = data.iloc[class_index].values

positive_coefficients = {k: v for k, v in zip(feature_names, coefficients) if v > 0}
positive_normalized_coefficients = {k: v / np.sum(list(positive_coefficients.values())) for k, v in positive_coefficients.items()}

positive_wordcloud = WordCloud(width = 950, height = 950,
                background_color ='skyblue',
                colormap= 'binary')
positive_wordcloud.generate_from_frequencies(positive_normalized_coefficients)

plt.figure(figsize=(10, 10))
plt.imshow(positive_wordcloud, interpolation="bilinear")

#Manchester United
class_index = 4

feature_names = data.columns.values
coefficients = data.iloc[class_index].values

positive_coefficients = {k: v for k, v in zip(feature_names, coefficients) if v > 0}
positive_normalized_coefficients = {k: v / np.sum(list(positive_coefficients.values())) for k, v in positive_coefficients.items()}

positive_wordcloud = WordCloud(width = 950, height = 950,
               background_color ='red',
                colormap= 'YlOrBr')
positive_wordcloud.generate_from_frequencies(positive_normalized_coefficients)

plt.figure(figsize=(10, 10))
plt.imshow(positive_wordcloud, interpolation="bilinear")

#Spurs
class_index = 5

feature_names = data.columns.values
coefficients = data.iloc[class_index].values

positive_coefficients = {k: v for k, v in zip(feature_names, coefficients) if v > 0}
positive_normalized_coefficients = {k: v / np.sum(list(positive_coefficients.values())) for k, v in positive_coefficients.items()}

positive_wordcloud = WordCloud(width = 950, height = 950,
                background_color ='white',
                colormap= 'Blues')
positive_wordcloud.generate_from_frequencies(positive_normalized_coefficients)

plt.figure(figsize=(10, 10))
plt.imshow(positive_wordcloud, interpolation="bilinear")

negative_coefficients = {k: v for k, v in zip(feature_names, coefficients) if v < 0}
negative_normalized_coefficients = {k: v / np.sum(list(negative_coefficients.values())) for k, v in negative_coefficients.items()}

negative_wordcloud = WordCloud(width=800, height = 800, background_color="white", colormap="coolwarm")
negative_wordcloud.generate_from_frequencies(negative_normalized_coefficients)

plt.figure(figsize=(10, 10))
plt.imshow(negative_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()