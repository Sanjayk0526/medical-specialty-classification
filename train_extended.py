# === Install required packages ===
# pip install -r requirements.txt

import kagglehub
import pandas as pd
import os
import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# === Step 1: Download MTSamples dataset from Kaggle ===
# Kaggle link: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
print("📥 Downloading MTSamples dataset...")
path = kagglehub.dataset_download("tboyle10/medicaltranscriptions")
print("Dataset downloaded to:", path)

mtsamples_file = os.path.join(path, "mtsamples.csv")
mtsamples_df = pd.read_csv(mtsamples_file)[['description', 'medical_specialty']].dropna()

# === Step 2: Load MEDIQA-AnS dataset from Hugging Face ===
# Hugging Face link: https://huggingface.co/datasets/mediqa
print("📥 Downloading MEDIQA-AnS dataset from Hugging Face...")
mediqa = load_dataset("mediqa", "AnS")
mediqa_df = pd.DataFrame(mediqa['train'])
mediqa_df = mediqa_df.rename(columns={
    'long_answer': 'description',
    'category': 'medical_specialty'
}).dropna()

# === Step 3: Merge datasets ===
combined_df = pd.concat([mtsamples_df, mediqa_df], ignore_index=True)
print(f"✅ Combined dataset size: {combined_df.shape[0]} samples")

# === Step 4: Clean text ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

combined_df['clean_desc'] = combined_df['description'].apply(clean_text)

# === Step 5: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    combined_df['clean_desc'], combined_df['medical_specialty'],
    test_size=0.2, random_state=42, stratify=combined_df['medical_specialty']
)

# === Step 6: TF-IDF vectorization ===
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# === Step 7: Train Logistic Regression ===
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)

# === Step 8: Evaluation ===
y_pred = clf.predict(X_test_tfidf)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# === Step 9: Save model and vectorizer ===
joblib.dump(clf, "extended_medical_specialty_model.joblib")
joblib.dump(vectorizer, "extended_tfidf_vectorizer.joblib")
print("💾 Model and vectorizer saved successfully!")
