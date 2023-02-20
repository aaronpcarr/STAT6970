from nltk.corpus import stopwords
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud

stop_words = stopwords.words('english')
stop_words.extend(['u','s','could','would','us'])

arsenal = pd.read_csv("Arsenal.csv")
arsenal = arsenal["Comment"]

arsenal_words = ""
# iterate through the csv file
for val in arsenal:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    arsenal_words += " ".join(tokens)+" "
 
arsenal_wordcloud = WordCloud(width = 950, height = 950,
                background_color ='red',
                colormap= 'Greys',
                stopwords= stop_words,
                min_font_size = 10).generate(arsenal_words)

plt.imshow(arsenal_wordcloud)

chelsea = pd.read_csv("Chelsea.csv")
chelsea = chelsea["Comment"]

chelsea_words = ""
for val in chelsea:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    chelsea_words += " ".join(tokens)+" "
 
chelsea_wordcloud = WordCloud(width = 950, height = 950,
                background_color ='blue',
                colormap= 'Greys',
                stopwords= stop_words,
                min_font_size = 10).generate(chelsea_words)

plt.imshow(chelsea_wordcloud)

liverpool = pd.read_csv("Liverpool.csv")
liverpool = liverpool["Comment"]

liverpool_words = ""
for val in liverpool:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    liverpool_words += " ".join(tokens)+" "
 
liverpool_wordcloud = WordCloud(width = 950, height = 950,
                background_color ='maroon',
                colormap= 'BuGn_r',
                stopwords= stop_words,
                min_font_size = 10).generate(liverpool_words)

plt.imshow(liverpool_wordcloud)

city = pd.read_csv("Manchester_City.csv")
city = city["Comment"]

city_words = ""
for val in city:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    city_words += " ".join(tokens)+" "
 
city_wordcloud = WordCloud(width = 950, height = 950,
                background_color ='skyblue',
                colormap= 'binary',
                stopwords= stop_words,
                min_font_size = 10).generate(city_words)

plt.imshow(city_wordcloud)

united = pd.read_csv("Manchester_United.csv")
united = united["Comment"]

united_words = ""
for val in united:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    united_words += " ".join(tokens)+" "
 
united_wordcloud = WordCloud(width = 950, height = 950,
                background_color ='red',
                colormap= 'YlOrBr',
                stopwords= stop_words,
                min_font_size = 10).generate(united_words)

plt.imshow(united_wordcloud)

spurs = pd.read_csv("Spurs.csv")
spurs = spurs["Comment"]

spurs_words = ""
for val in spurs:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    spurs_words += " ".join(tokens)+" "
 
spurs_wordcloud = WordCloud(width = 950, height = 950,
                background_color ='white',
                colormap= 'Blues',
                stopwords= stop_words,
                min_font_size = 10).generate(spurs_words)

plt.imshow(spurs_wordcloud)
