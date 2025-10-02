# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

# ต้อง import โมดูล tokenizer ไว้ให้ joblib เห็นตอน unpickle
# (ถึงแม้เราโหลด vectorizer ที่ fit แล้ว แต่ถ้าในนั้นอ้างถึงฟังก์ชัน ต้องให้ path นี้มีตัวตน)
from thai_tokenizer import thai_tokenizer  # noqa: F401  (ใช้ให้มีใน namespace)

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401

app = FastAPI(title="Thai To-do/Summarize Classifier")

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.joblib")
VECT_PATH  = os.path.join(MODEL_DIR, "vectorizer.joblib")

if not (os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH)):
    raise SystemExit("❌ ไม่พบไฟล์โมเดล/เวกเตอร์ไรเซอร์ กรุณา run `python train.py` ก่อน")

# โหลดของที่ fit แล้ว (มี vocab + idf ครบ)
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

class TextRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: TextRequest):
    try:
        X = vectorizer.transform([req.text])
        pred = model.predict(X)[0]

        # ความน่าจะเป็นต่อคลาส (ถ้ามี)
        proba_map = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            labels = list(model.classes_)
            proba_map = {labels[i]: float(probs[i]) for i in range(len(labels))}

        return {"input": req.text, "prediction": pred, "probability": proba_map}
    except Exception as e:
        # โยนเป็น 500 ให้เห็นชัดเจนตอน dev
        raise HTTPException(status_code=500, detail=str(e))
