import pandas as pd
import ast
import matplotlib.pyplot as plt
import networkx as nx

dataset = pd.read_csv('tweets.csv')

print('Dataset Information :- ')
print(dataset.info())
print()

print('Dataset Statistics :- ')
print(dataset.describe())
print()

print('After Removing Missing Values :- ')
dataset = dataset[dataset.user_location.notnull()][['user','tweet','additional_data']]
print(dataset)

dataset['City'] = ''
dataset['Country'] = ''
dataset['hashtags'] = ''

for i,row in dataset.iterrows():
    dataset['City'][i] = ast.literal_eval(row.additional_data)['place']['name']
    dataset['Country'][i] = ast.literal_eval(row.additional_data)['place']['country']
    hashtags = [token for token in row.tweet.split() if token.startswith('#')]
    dataset['hashtags'][i] = hashtags

dataset['Country'].value_counts().plot(kind='barh')
plt.xlabel('Tweet Count')
plt.ylabel('Country')
plt.title('Tweets per Country')
plt.show()

dataset['City'].value_counts().plot(kind='pie')
plt.title('Tweets per City')
plt.legend()
plt.show()

dataset = dataset[:10]
dataset = dataset.explode('hashtags')
users = list(dataset['user'].unique())
hashtags = list(dataset['hashtags'].unique())

vis = nx.Graph()
vis.add_nodes_from(users + hashtags)

for name,group in dataset.groupby(['hashtags','user']):
    hashtag, user = name
    weight = len(group)
    vis.add_edge(hashtag,user, weight=weight)

nx.draw(vis, with_labels=True)
plt.show()
