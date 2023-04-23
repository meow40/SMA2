import pandas as pd
import ast
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

dataset = pd.read_csv('tweets.csv')
dataset['Country'] = 'None'
dataset['City'] = 'None'

for i,row in dataset.iterrows():
    loc_data = ast.literal_eval(row['additional_data'])
    if loc_data['place']:
        dataset['Country'][i] = loc_data['place']['country']
        dataset['City'][i] = loc_data['place']['name']

dataset = dataset[dataset['City'] != 'None']
print(dataset[['Country','City']])
print()

city_frequency = dataset['City'].value_counts()
print('City Frequency :- ')
print(city_frequency)
print()

country_frequency = dataset['Country'].value_counts()
print('Country Frequency :- ')
print(country_frequency)
print()

# Plot Top 15
plt.bar(city_frequency.index[:15],city_frequency.values[:15])
plt.title('City Frequency')
plt.xlabel('City')
plt.ylabel('Tweet Count')
plt.show()

plt.bar(country_frequency.index[:15],country_frequency.values[:15])
plt.xlabel('Country')
plt.ylabel('Tweet Count')
plt.title('Country Frequency')
plt.show()
