import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from wordcloud import WordCloud
import warnings
import numpy as np
warnings.filterwarnings('ignore')

stops = stopwords.words('english')
lem = WordNetLemmatizer()

swig = pd.read_csv('swiggy.csv')[:500]
zoma = pd.read_csv('zomato.csv')[:500]

# Compare Retweet Count
swig_rt = swig['retweet_count'].sum()
zoma_rt = zoma['retweet_count'].sum()
plt.bar(['Swiggy','Zomato'],[swig_rt,zoma_rt])
plt.xlabel('Company')
plt.ylabel('No. of Retweets')
plt.title('Retweet Counts of Swiggy and Zomato Tweets')
plt.show()

# Compare Tweet Length
swig_len = swig['length'].sum()/len(swig['length'])
zoma_len = zoma['length'].sum()/len(zoma['length'])
plt.bar(['Swiggy','Zomato'],[swig_len,zoma_len])
plt.xlabel('Company')
plt.ylabel('Average Tweet Length')
plt.title('Average Tweet Length of Swiggy and Zomato Tweets')
plt.show()

# Compare Favorite Count
swig_fav = swig['favorite_count'].sum()
zoma_fav = zoma['favorite_count'].sum()
plt.bar(['Swiggy','Zomato'],[swig_fav,zoma_fav])
plt.xlabel('Company')
plt.ylabel('Favorite Count')
plt.title('Favorite Counts of Swiggy and Zomato Tweets')
plt.show()

# Compare Trends
def analyze_trends(dataset, company):
    dataset['created_at'] = pd.to_datetime(dataset['created_at'])
    dataset['date'] = dataset['created_at'].dt.date
    dataset['hour'] = dataset['created_at'].dt.hour

    date_freq = dataset['date'].value_counts()
    hour_freq = dataset['hour'].value_counts()

    plt.bar(hour_freq.index, hour_freq.values)
    plt.xlabel('Hour of day when tweeted')
    plt.ylabel('No. of Tweets')
    plt.title(f'Hourly Tweet Activity of {company} tweets')
    plt.show()

    x = date_freq.index
    x_num = dates.date2num(x)
    y = date_freq.values
    plt.scatter(x,y)
    z = np.polyfit(x_num, y, deg=1)
    p = np.poly1d(z)
    plt.plot(x, p(x_num))

    plt.xlabel('Date when tweeted')
    plt.ylabel('No. of Tweets')
    plt.title(f'Tweet Trend of {company} tweets')
    plt.show()

analyze_trends(zoma,'Zomato')
analyze_trends(swig,'Swiggy')

# Compare Sentiment
def analyze_sentiment(dataset, company):
    dataset['polarity'] = ''
    dataset['sentiment'] = ''
    corpus = ''

    for i,row in dataset.iterrows():
        row_tokens = word_tokenize(row.full_text.lower())
        row_tokens = [token for token in row_tokens if not token.startswith('http')]
        row_tokens = [token for token in row_tokens if token not in stops]
        row_tokens = [token for token in row_tokens if token.isalpha()]
        row_tokens = [lem.lemmatize(token) for token in row_tokens]

        row_tokens = ' '.join(row_tokens)
        corpus += row_tokens + ' '

        senti = TextBlob(row_tokens)
        dataset['polarity'][i] = senti.sentiment.polarity
        if dataset['polarity'][i] > 0:
            dataset['sentiment'][i] = 'positive'
        elif dataset['polarity'][i] < 0:
            dataset['sentiment'][i] = 'negative'
        else:
            dataset['sentiment'][i] = 'neutral'

    dataset['sentiment'].value_counts().plot(kind='barh')
    plt.xlabel('Tweet Count')
    plt.ylabel('Sentiment')
    plt.title(f'Sentiment analysis of {company} Tweets')
    plt.show()

    return corpus

zoma_corpus = analyze_sentiment(zoma,'Zomato')
swig_corpus = analyze_sentiment(swig,'Swiggy')

# Compare Word Cloud
def generate_wordcloud(corpus, company):
    cloud = WordCloud(background_color='white', width=800, height=800, max_words=50).generate(corpus)
    plt.imshow(cloud)
    plt.axis('off')
    plt.title(f'Word Cloud for what customers are saying about {company} :-')
    plt.show()

generate_wordcloud(zoma_corpus,'Zomato')
generate_wordcloud(swig_corpus, 'Swiggy')
