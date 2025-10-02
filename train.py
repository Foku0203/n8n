# train.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib
from thai_tokenizer import thai_tokenizer  # <-- อ้างจากไฟล์แยก

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

# ลบแถวว่าง (ถ้ามี)
df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

X = df["text"].astype(str)
y = df["label"].astype(str)

# TF-IDF ที่ใช้ tokenizer ภาษาไทย
# ตั้ง token_pattern=None เพื่อไม่ให้ scikit-learnเตือน และ lowercase=False เผื่อภาษาไทย
vectorizer = TfidfVectorizer(
    tokenizer=thai_tokenizer,
    token_pattern=None,
    lowercase=False,
)

X_vec = vectorizer.fit_transform(X)

# โมเดลง่ายๆ SVM linear
model = SVC(kernel="linear", probability=True, random_state=42)
model.fit(X_vec, y)

# เซฟ "โมเดล" และ "vectorizer ที่ fit แล้ว" (มี vocab+idf ครบ)
joblib.dump(model, os.path.join(MODEL_DIR, "svm_model.joblib"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))

print("✅ Training เสร็จ! เซฟไว้ที่ model/svm_model.joblib และ model/vectorizer.joblib")
