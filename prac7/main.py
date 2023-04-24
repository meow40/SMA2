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
dataset = dataset[dataset['user_location'].isnull() == False][['tweet','additional_data']]
print(dataset)

dataset['City'] = ''
dataset['Country'] = ''

for i,row in dataset.iterrows():
    dataset['City'][i] = ast.literal_eval(row.additional_data)['place']['name']
    dataset['Country'][i] = ast.literal_eval(row.additional_data)['place']['country']

dataset['Country'].value_counts().plot(kind='barh')
plt.xlabel('Tweet Count')
plt.ylabel('Country')
plt.title('Tweets per Country')
plt.show()

dataset['City'].value_counts().plot(kind='pie')
plt.title('Tweets per City')
plt.legend()
plt.show()


vis = nx.Graph()
countries = list(dataset['Country'].unique())
cities = list(dataset['City'].unique())
vis.add_nodes_from(countries+cities)

for name, group in dataset.groupby(['Country','City']):
    country, city = name
    tweet_count = len(group)
    vis.add_edge(country, city, weight=tweet_count, title=tweet_count, label=f'Tweets : {tweet_count}')

nx.draw(vis, with_labels=True)
plt.show()
