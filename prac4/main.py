import pandas as pd
import ast
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

dataset = pd.read_csv('tweets.csv')
dataset['Country'] = 'None'
dataset['Hashtags'] = ''

for i,row in dataset.iterrows():
    loc_data = ast.literal_eval(row['additional_data'])
    hashtags = [token for token in row.tweet.split() if token.startswith('#')]
    dataset['Hashtags'][i] = hashtags

    if loc_data['place']:
        dataset['Country'][i] = loc_data['place']['country']

dataset = dataset[dataset['Country'] != 'None']
dataset = dataset[['Country','Hashtags']]
dataset = dataset.explode('Hashtags')

for country in dataset['Country'].unique():
    dataset_country = dataset[dataset['Country'] == country]
    dataset_country['Hashtags'].value_counts().plot(kind = 'barh')
    plt.xlabel('Hashtag Count')
    plt.ylabel('Hashtags')
    plt.title(f'Hashtag Popularity in {country}')
    plt.show()
