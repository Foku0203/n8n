from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import threading
from datetime import datetime

# ให้ joblib เห็นสัญลักษณ์ที่ใช้ตอน pickle
from thai_tokenizer import thai_tokenizer  # noqa: F401
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401

app = FastAPI(title="Thai To-do/Summarize Classifier")

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.joblib")
VECT_PATH  = os.path.join(MODEL_DIR, "vectorizer.joblib")
ID_STATE_PATH = os.path.join(MODEL_DIR, "last_id.txt")

if not (os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH)):
    raise SystemExit("❌ ไม่พบไฟล์โมเดล/เวกเตอร์ไรเซอร์ กรุณา run `python train.py` ก่อน")

# โหลดโมเดล + เวกเตอร์ไรเซอร์
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

# ===== ID state =====
id_lock = threading.Lock()

def _load_last_id() -> int:
    try:
        with open(ID_STATE_PATH, "r", encoding="utf-8") as f:
            return int(f.read().strip() or "0")
    except:
        return 0

def _save_last_id(v: int) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(ID_STATE_PATH, "w", encoding="utf-8") as f:
        f.write(str(v))

last_id = _load_last_id()
# ====================

class TextRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: TextRequest):
    global last_id
    try:
        text = (req.text or "").strip()
        if not text:
            raise HTTPException(status_code=422, detail="text ว่าง")

        # แปลงข้อความเป็นเวกเตอร์
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]

        # ความน่าจะเป็นของแต่ละ class
        prob_other, prob_sum, prob_todo = None, None, None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            labels = list(model.classes_)
            prob_map = {labels[i]: float(probs[i]) for i in range(len(labels))}
            prob_other = prob_map.get("Other")
            prob_sum   = prob_map.get("Summarize")
            prob_todo  = prob_map.get("To-do")

        # Auto-increment id
        with id_lock:
            last_id += 1
            _save_last_id(last_id)
            new_id = last_id

        # timestamp ISO format
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "id": new_id,
            "text": text,
            "label": pred,
            "prob_other": prob_other,
            "prob_summarize": prob_sum,
            "prob_todo": prob_todo,
            "source": "line_export",
            "created_at": timestamp
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
