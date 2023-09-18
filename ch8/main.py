import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.porter import PorterStemmer

count = CountVectorizer()
docs = np.array(
    [
        "The sun is shining",
        "The weather is sweet",
        "The sun is shining, the weather is sweet",
        "and one and one is two",
    ]
)
bag = count.fit_transform(docs)
print(bag.toarray())

tfidf = TfidfTransformer(use_idf=True, norm="l2", smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

df = pd.read_csv("movie_data.csv")


def clean_text(text: str):
    removeme = [
        ":)",
        ":/",
        ":-)" ":D",
        ":|",
        ":(",
        ":P",
        ";)",
        ";/",
        ";-)" ";D",
        ";|",
        ";(" ";P",
        "=)",
        "=/",
        "=-)" "=D",
        "=|",
        "=(" "=P",
        "<br />",
        "<br/>",
    ]
    out = text
    for r in removeme:
        out = out.replace(r, "")
    return out


from typing import List


def tokenizer(text: str) -> List[str]:
    return text.split()


porter = PorterStemmer()


def tokenizer_porter(text: str) -> List[str]:
    return [porter.stem(word) for word in text.split()]


df["review"] = df["review"].apply(clean_text)

import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

stop = stopwords.words("english")

result = [
    w for w in tokenizer("a runner likes running and runs a lot") if w not in stop
]
print(result)

X_train = df.loc[:25000, "review"].values
y_train = df.loc[:25000, "sentiment"].values
X_test = df.loc[25000:, "review"].values
y_test = df.loc[25000:, "sentiment"].values

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def inmemory_regression_model():
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    small_param_grid = [
        {
            "vect__ngram_range": [(1, 1)],
            "vect__stop_words": [None],
            "vect__tokenizer": [tokenizer, tokenizer_porter],
            "clf__penalty": ["l2"],
            "clf__C": [1.0, 10.0],
        },
        {
            "vect__ngram_range": [(1, 1)],
            "vect__stop_words": [stop, None],
            "vect__tokenizer": [tokenizer],
            "vect__use_idf": [False],
            "vect__norm": [None],
            "clf__penalty": ["l2"],
            "clf__C": [1.0, 10.0],
        },
    ]

    lr_tfidf = Pipeline(
        [("vect", tfidf), ("clf", LogisticRegression(solver="liblinear"))]
    )

    gs_lr_tfidf = GridSearchCV(
        lr_tfidf, small_param_grid, scoring="accuracy", cv=5, verbose=2, n_jobs=-1
    )

    gs_lr_tfidf.fit(X_train, y_train)

    print(f"Best parameter set: {gs_lr_tfidf.best_params_}")
    print(f"CV score: {gs_lr_tfidf.best_score_}")
    clf = gs_lr_tfidf.best_estimator_
    print(f"Test accuracy: {clf.score(X_test, y_test):.3f}")


def stream_docs(path):
    with open(path, "r", encoding="utf8") as csv:
        next(csv)
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


def outofcore_example():
    import pyprind

    vect = HashingVectorizer(
        decode_error="ignore",
        n_features=2**21,
        preprocessor=None,
        tokenizer=tokenizer,
    )

    clf = SGDClassifier(loss="log_loss", random_state=1)
    doc_stream = stream_docs(path="movie_data.csv")

    pbar = pyprind.ProgBar(45)
    classes = np.array([0, 1])
    for _ in range(45):
        X_train, y_train = get_minibatch(doc_stream, size=1000)
        if not X_train:
            break
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=classes)
        pbar.update()

    X_test, y_test = get_minibatch(doc_stream=doc_stream, size=5000)
    X_test = vect.transform(X_test)
    print(f"Accuracy: {clf.score(X_test, y_test)}")

    clf.partial_fit(X_test, y_test)
