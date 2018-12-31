import pandas as pd

df = pd.read_csv('my_data.csv')
df.drop(['entry_id', 'pos1', 'pos2'], axis=1, inplace=True)

print(len(df))

x = df['bread1'].values.tolist()
y = df['bread2'].values.tolist()
corpus = x + y

print(len(corpus))

import re
import nltk

nltk.download('stopwords')
sw_list = nltk.corpus.stopwords.words('english')


# removing all the special characters
def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist


def preprocess(x):
    s = ''
    x = re.sub('\W+', ' ', x)
    for data in x:
        if data not in sw_list:
            s += " " + data
    s = s[1:]
    s = '  '.join(unique_list(s.split()))
    return x


new_corpus = []
for data in corpus:
    data = str(data).lower()
    new_corpus.append(preprocess(data))

corpus = new_corpus

print(corpus[0])

# copy data in "latest_file.csv"
with open('latest_file.csv', 'w') as f:
    i = 0
    for data in corpus:
        if i < len(corpus) / 10000:
            f.write("%s\n" % data)
            i += 1

df = pd.read_csv('latest_file.csv', names=['breads', 'label'])

df.head()
