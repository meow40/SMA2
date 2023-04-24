import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

dataset = pd.read_csv('tweets.csv')

most_retweeted = dataset.sort_values(by='retweet_count', ascending=False)
most_retweeted = most_retweeted[['retweet_count','tweet']]
print('Top 5 Most Retweeted Tweets :- ')

for i,row in most_retweeted.head(5).iterrows():
    print('-------------------------------------------------')
    print(f'Tweet : {row.tweet}')
    print(f'Retweet Count : {row.retweet_count}')
    print('-------------------------------------------------')
    print()

dataset['hashtags'] = ''
for i,row in dataset.iterrows():
    tokens = row.tweet.split()
    hashtags = [token for token in tokens if token.startswith('#')]
    dataset['hashtags'][i] = hashtags

dataset = dataset[['retweet_count','hashtags']]
dataset = dataset.explode('hashtags')
retweeted_ht = dataset.groupby('hashtags')['retweet_count'].sum()
most_retweeted_ht = retweeted_ht.sort_values(ascending=False)
print()
print('Top 10 Most Retweeted Hashtags :- ')
print(most_retweeted_ht.head(10))
most_retweeted_ht.head(10).plot(kind='barh')
plt.xlabel('Retweet Count')
plt.ylabel('Hashtag')
plt.title('Most Retweeted Hashtags')
plt.show()
