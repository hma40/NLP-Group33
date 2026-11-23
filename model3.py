from pathlib import Path
import re
import numpy as np
import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #install VADER to use this

#add "--dataset (name of data set) --split (dev/test)" to choose data set
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=[
        "semeval2018",
        "semeval2022",
        "figlang2020/twitter",
        "figlang2020/reddit",
        "sarc",
        "nikesh66",
        "hahackathon",
        "semeval2022_idioms",
    ],
    default="semeval2018",
)
parser.add_argument("--split", choices=["dev", "test"], default="dev")
args = parser.parse_args()
#each dataset uses its own train, development, and test files.
data_dir = Path("Data/Processed") / args.dataset
print(f"Dataset: {args.dataset}")
print(f"Evaluation split: {args.split}")

train = pd.read_csv(data_dir / "train.csv", keep_default_na=False)
dev = pd.read_csv(data_dir / f"{args.split}.csv", keep_default_na=False)

X_train, y_train = train["text"], train["label"]
X_dev, y_dev = dev["text"], dev["label"]

#extract six style features from each text: word count, counts of ! and ?, 
#repeated punctuation, all-caps word count, and presence of double quotes.
def extract_style_features(texts):
    rows = []

    for text in texts:
        words = re.findall(r"\b[^\W\d_]+\b", text)
        all_caps = sum(len(word) > 1 and word.isupper() for word in words)

        rows.append([
            len(words),                               
            text.count("!"),                          
            text.count("?"),                          
            int(bool(re.search(r"[!?]{2,}", text))),  
            all_caps,                                 
            int(any(mark in text for mark in '"“”')), 
        ])

    return np.asarray(rows, dtype=float)

#initialize VADER once and reuse it for all texts
sentiment_analyzer = SentimentIntensityAnalyzer()

#extract four VADER scores per text: positive, negative, neutral, and compound.
#add a binary flag when both positive and negative scores are greater than zero.
def extract_sentiment_features(texts):
    rows = []

    for text in texts:
        scores = sentiment_analyzer.polarity_scores(text)

        # Flag texts where VADER detects both positive and negative sentiment.
        mixed_sentiment = int(scores["pos"] > 0 and scores["neg"] > 0)

        rows.append([
            scores["pos"],
            scores["neg"],
            scores["neu"],
            scores["compound"],
            mixed_sentiment,
        ])

    return np.asarray(rows, dtype=float)

#split on contrast words such as "but", "yet", and "though".
CONTRAST_WORDS = re.compile(r"\b(?:but|yet|though)\b", re.IGNORECASE)
def extract_clause_contrast_features(texts):
    rows = []
    for text in texts:
        parts = CONTRAST_WORDS.split(text)
        max_gap = 0.0
        polarity_flip = 0
        #score each nonempty text segment once
        scores = []
        for part in parts:
            if part.strip():
                scores.append(sentiment_analyzer.polarity_scores(part)["compound"])
            else:
                scores.append(None)
        #compare the segments immediately before and after each contrast word
        for left_score, right_score in zip(scores, scores[1:]):
            if left_score is None or right_score is None:
                continue
            #measure how far apart the two sentiment scores are
            max_gap = max(max_gap, abs(left_score - right_score))
            #require one positive side and one negative side
            if min(left_score, right_score) <= -0.05 and max(left_score, right_score) >= 0.05:
                polarity_flip = 1
        # both features are zero when there is no usable pair
        rows.append([max_gap, polarity_flip])

    return np.asarray(rows, dtype=float)

#append the scaled style features to each text's TF-IDF vector.
#the scaler learns its mean and standard deviation from training data.
features = FeatureUnion([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ("style", make_pipeline(
        FunctionTransformer(extract_style_features),
        StandardScaler(),
    )),
    #scale sentiment features using training-set statistics
    ("sentiment", make_pipeline(
        FunctionTransformer(extract_sentiment_features),
        StandardScaler(),
    )),
    #add clause-level sentiment contrast on top of sentiment features
    ("clause_contrast", make_pipeline(
        FunctionTransformer(extract_clause_contrast_features),
        StandardScaler(),
    )),
])

#then use the same TF-IDF and logistic regression settings as Model 2
#to measure the effect of adding style features on the same dev split.
model = make_pipeline(
    features,
    LogisticRegression(C=1.0, max_iter=1000, class_weight = "balanced")
)

model.fit(X_train, y_train)
dev_pred = model.predict(X_dev)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_dev, dev_pred, average="binary", pos_label=1, zero_division=0
)

tn, fp, fn, tp = confusion_matrix(y_dev, dev_pred, labels=[0, 1]).ravel()

print(f"Train size:         {len(train)}")
print(f"Dev size:           {len(dev)}")
print(f"Accuracy:           {accuracy_score(y_dev, dev_pred):.4f}")
print(f"Positive precision: {precision:.4f}")
print(f"Positive recall:    {recall:.4f}")
print(f"Positive F1:        {f1:.4f}")
print(f"Irony found:        {tp}/{tp + fn}")
print(f"False positives:    {fp}")