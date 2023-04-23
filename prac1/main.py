import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim
import pyLDAvis
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('tweets.csv')
tokens = []
stops = stopwords.words('english')
lem = WordNetLemmatizer()

for i,row in dataset.iterrows():
    row_tokens = word_tokenize(row.tweet.lower())
    row_tokens = [token for token in row_tokens if not token.startswith('http')]
    row_tokens = [token for token in row_tokens if token not in stops]
    row_tokens = [token for token in row_tokens if token.isalpha()]
    row_tokens = [lem.lemmatize(token) for token in row_tokens]
    tokens.append(row_tokens)

dictionary = Dictionary(tokens) # wordid -> word mapping
corpus = [dictionary.doc2bow(row_tokens) for row_tokens in tokens] # wordid -> frequency mapping
lda = LdaModel(corpus = corpus, num_topics = 10, id2word=dictionary, passes=10, random_state=13)

print('Following topics were extracted from the tweets : ')
for i,topic in lda.show_topics():  # prints keywords in topic and their weights
    print(f'Topic {i}: {topic}')

print()
print("LDA Model Perplexity: ", lda.log_perplexity(corpus)) # Lower the perplexity, better the model

vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(vis,'lda_vis.html')
