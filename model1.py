from pathlib import Path
import pandas as pd
import argparse

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

y_train = pd.read_csv(data_dir / "train.csv", usecols=["label"])["label"]
y_dev = pd.read_csv(data_dir / f"{args.split}.csv", usecols=["label"])["label"]

label_counts = y_train.value_counts().sort_index()
majority_label = label_counts.idxmax()

print("Train size:", len(y_train))
print("Dev size:", len(y_dev))
print("Training label counts:")
print(label_counts)
print("Majority label:", majority_label)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

dev_pred = [majority_label] * len(y_dev)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_dev,
    dev_pred,
    average="binary",
    pos_label=1,
    zero_division=0,
)

print(f"Accuracy:           {accuracy_score(y_dev, dev_pred):.4f}")
print(f"Positive precision: {precision:.4f}")
print(f"Positive recall:    {recall:.4f}")
print(f"Positive F1:        {f1:.4f}")