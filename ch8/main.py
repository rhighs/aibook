import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.porter import PorterStemmer

count = CountVectorizer()
docs = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining, the weather is sweet',
                 'and one and one is two'])
bag = count.fit_transform(docs)
print(bag.toarray())

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

df = pd.read_csv('movie_data.csv')

def clean_text(text: str):
    removeme = [
        ':)', ':/', ':-)'
        ':D', ':|', ':(',
        ':P',
        ';)', ';/', ';-)'
        ';D', ';|', ';('
        ';P',
        '=)', '=/', '=-)'
        '=D', '=|', '=('
        '=P',
        '<br />', '<br/>'
    ]
    out = text
    for r in removeme:
        out = out.replace(r, '')
    return out

from typing import List
def tokenizer(text: str) -> List[str]:
    return text.split()

porter = PorterStemmer()
def tokenizer_porter(text: str) -> List[str]:
    return [porter.stem(word) for word in text.split()]

df['review'] = df['review'].apply(clean_text)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

result = [w for w in tokenizer('a runner likes running and runs a lot') if w not in stop]
print(result)

