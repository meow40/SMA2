import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

dataset = pd.read_csv('tweets.csv')
stops = stopwords.words('english')
lem = WordNetLemmatizer()

dataset['sentiment'] = ''
dataset['polarity'] = 0.0

for i,row in dataset.iterrows():
    row_tokens = word_tokenize(row.tweet.lower())
    row_tokens = [token for token in row_tokens if not token.startswith('http')]
    row_tokens = [token for token in row_tokens if token not in stops]
    row_tokens = [token for token in row_tokens if token.isalpha()]
    row_tokens = [lem.lemmatize(token) for token in row_tokens]

    senti = TextBlob(' '.join(row_tokens))
    dataset['polarity'][i] = senti.sentiment.polarity
    if dataset['polarity'][i] > 0:
        dataset['sentiment'][i] = 'positive'
    elif dataset['polarity'][i] < 0:
        dataset['sentiment'][i] = 'negative'
    else:
        dataset['sentiment'][i] = 'neutral'

print('Sentiments Counts :- ')
print(dataset['sentiment'].value_counts())
sentiment_freq = dataset['sentiment'].value_counts().plot(kind='barh')
plt.xlabel('Tweet Count')
plt.ylabel('Sentiment')
plt.title('Sentiment Analysis of Tweets')
plt.show()


# Plot the distribution of polarity scores
sns.kdeplot(dataset[dataset['sentiment'] == 'positive']['polarity'], shade=True, label = 'Positive')
sns.kdeplot(dataset[dataset['sentiment'] == 'negative']['polarity'], shade=True, label = 'Negative')
plt.xlabel('Polarity')
plt.ylabel('Density')
plt.title('Sentiment Analysis')
plt.legend()
plt.show()
