import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('tweets.csv')
dataset = dataset[['created_at','tweet']]
dataset['created_at'] = pd.to_datetime(dataset['created_at'])

dataset['date'] = dataset['created_at'].dt.date
dataset['hour'] = dataset['created_at'].dt.hour

date_freq = dataset['date'].value_counts()
print('Date Frequency :- ')
print(date_freq)
print()

hour_freq = dataset['hour'].value_counts()
print('Hour Frequency :- ')
print(hour_freq)
print()

plt.bar(hour_freq.index,hour_freq.values)
plt.xlabel('Hour of day when tweeted')
plt.ylabel('No. of Tweets')
plt.title('Tweet Activity based on hour of day')
plt.show()

plt.plot(date_freq.index,date_freq.values, '-o')
plt.xlabel('Date when tweeted')
plt.ylabel('No. of Tweets')
plt.title('Tweet Trend')
plt.show()
