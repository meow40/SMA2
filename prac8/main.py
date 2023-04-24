import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('reviews.csv')
stops = stopwords.words('english')
lem = WordNetLemmatizer()
dataset['cleaned_review'] = ''
dataset['sentiment'] = ''

corpus = ''

for i,row in dataset.iterrows():
    review = [token for token in word_tokenize(str(row.review_text).lower())]
    review = [token for token in review if token not in stops]
    review = [lem.lemmatize(token) for token in review if token not in stops]
    review = [token.translate(str.maketrans('','',string.punctuation)) for token in review]
    review = ' '.join(review)
    dataset['cleaned_review'][i] = review
    senti = TextBlob(review)
    senti = senti.sentiment.polarity
    if senti > 0:
        dataset['sentiment'][i] = 'positive'
    elif senti < 0:
        dataset['sentiment'][i] = 'negative'
    else:
        dataset['sentiment'][i] = 'neutral'

    corpus += review + ' '

cloud = WordCloud(background_color='white',width=800,height=800, max_words=50).generate(corpus)
plt.imshow(cloud)
plt.axis('off')
plt.title('Word Cloud for what customers are saying about Apple Iphone :- ')
plt.show()

dataset = dataset[['cleaned_review','sentiment']]
dataset['sentiment'].value_counts().plot(kind='barh')
plt.xlabel('Count')
plt.ylabel('Sentiment')
plt.title('Sentiment of Reviews of IPhone')
plt.show()
