import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
try:
    df = pd.read_csv("mail_data.csv", encoding="ISO-8859-1")
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: The file 'mail_data.csv' was not found.")
    exit()
df.rename(columns={"Category": "label", "Message": "text"}, inplace=True)
df["label"] = df["label"].map({"spam": 1, "ham": 0})
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["text"] = df["text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(min_df=1, stop_words="english", binary=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_features, y_train)

train_predictions = model.predict(X_train_features)
train_accuracy = accuracy_score(y_train, train_predictions)

test_predictions = model.predict(X_test_features)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"✅ Accuracy on Training Data: {train_accuracy:.4f}")
print(f"✅ Accuracy on Testing Data: {test_accuracy:.4f}")
