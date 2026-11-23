from pathlib import Path
import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline

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

#use the processed text as input and the binary irony label as the target
X_train, y_train = train["text"], train["label"]
X_dev, y_dev = dev["text"], dev["label"]

#use TF-IDF with unigrams and bigrams. Keep terms that appear in at least 
#two training documents, then train logistic regression.
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2),
    LogisticRegression(C=1.0, max_iter=1000, class_weight = "balanced")
)

#learn the vocabulary, IDF weights, and classifier from training data, and 
#apply the fitted pipeline to the development split.
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