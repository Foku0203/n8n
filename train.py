# train.py
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from thai_tokenizer import thai_tokenizer  # tokenizer แยกคำภาษาไทย

# ---------------------------
# 1. Path & Dataset
# ---------------------------
DATA_PATH = "data/mockdata.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# อ่าน CSV กัน BOM/space/คอมม่าท้าย header
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", skipinitialspace=True)
df.columns = df.columns.str.replace("\ufeff", "", regex=True).str.strip().str.lower()

# เช็คคอลัมน์จำเป็น
required = {"text", "label"}
if not required.issubset(df.columns):
    raise SystemExit(f"CSV ต้องมีคอลัมน์ {required} แต่พบ {list(df.columns)}")

# ลบแถวว่าง
df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

X = df["text"].astype(str)
y = df["label"].astype(str)

# ---------------------------
# 2. Train/Test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 3. Vectorizer + Tokenizer ภาษาไทย
# ---------------------------
vectorizer = TfidfVectorizer(
    tokenizer=thai_tokenizer,
    token_pattern=None,
    lowercase=False,
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------
# 4. Train Model (SVM Linear)
# ---------------------------
model = SVC(kernel="linear", probability=True, random_state=42)
model.fit(X_train_vec, y_train)

# ---------------------------
# 5. Evaluate Model
# ---------------------------
y_pred = model.predict(X_test_vec)
print("📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📑 Classification Report:")
print(classification_report(y_test, y_pred, digits=3))

# ---------------------------
# 6. Save Model & Vectorizer
# ---------------------------
joblib.dump(model, os.path.join(MODEL_DIR, "svm_model.joblib"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))

print("\n✅ Training เสร็จ! เซฟไว้ที่:")
print("   - model/svm_model.joblib")
print("   - model/vectorizer.joblib")
