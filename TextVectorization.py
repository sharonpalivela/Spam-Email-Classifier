
import pandas as pd
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. Load the dataset
try:
    df = pd.read_csv("mail_data.csv", encoding="ISO-8859-1")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: The file 'mail_data.csv' was not found.")
    raise

# 2. Rename columns
df.rename(
    columns={"Category": "label", "Message": "text"},
    inplace=True
)

# 3. Convert labels into numbers
df["label"] = df["label"].map({"spam": 1, "ham": 0})

# Remove rows with missing or unrecognized labels/text
df.dropna(subset=["label", "text"], inplace=True)

# 4. Clean the email text
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(clean_text)

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print(f"\nTotal emails: {len(df)}")
print(f"Training emails: {len(X_train)}")
print(f"Testing emails: {len(X_test)}")

# 6. Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    min_df=1,
    stop_words="english",
    binary=True
)

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# 7. Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, y_train)

# 8. Make predictions
train_predictions = model.predict(X_train_features)
test_predictions = model.predict(X_test_features)

# 9. Calculate training and testing accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

# 10. Calculate evaluation metrics for the spam class
precision = precision_score(y_test, test_predictions, zero_division=0)
recall = recall_score(y_test, test_predictions, zero_division=0)
f1 = f1_score(y_test, test_predictions, zero_division=0)

# 11. Display results
print("\n========== MODEL EVALUATION ==========")

print(f"Training Accuracy: {train_accuracy:.2%}")
print(f"Testing Accuracy:  {test_accuracy:.2%}")
print(f"Spam Precision:    {precision:.4f}")
print(f"Spam Recall:       {recall:.4f}")
print(f"Spam F1-Score:     {f1:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Ham", "Spam"],
        zero_division=0
    )
)
