import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ================= CLEAN TEXT FUNCTION =================
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ================= LOAD DATA =================
df = pd.read_csv(r"C:\\Users\\User\\Downloads\\archive (3)\\training_data.csv")

print("Columns:", df.columns)

# ================= FIX COLUMN NAMES =================
df.columns = df.columns.str.strip().str.lower()

# expected columns
# resume text, skills, education, category

# ================= CHECK REQUIRED =================
required_cols = ['resume text', 'skills', 'education', 'category']

for col in required_cols:
    if col not in df.columns:
        raise Exception(f"{col} column not found")

# ================= COMBINE TEXT =================
df['full_text'] = (
    df['resume text'].fillna('') + " " +
    df['skills'].fillna('') + " " +
    df['education'].fillna('')
)

# ================= CLEAN =================
df['full_text'] = df['full_text'].apply(clean_text)

# ================= REMOVE EMPTY ROWS =================
df = df[df['full_text'].str.strip() != '']

print("Remaining rows after cleaning:", len(df))

# ================= FEATURES =================
X = df['full_text']
y = df['category']

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= TF-IDF =================
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ================= MODEL =================
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ================= PREDICT =================
y_pred = model.predict(X_test_vec)

# ================= EVALUATION =================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ================= SAVE =================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel & Vectorizer Saved Successfully!")