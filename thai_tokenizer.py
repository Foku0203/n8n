# thai_tokenizer.py
from pythainlp.tokenize import word_tokenize

def thai_tokenizer(text: str):
    if not isinstance(text, str):
        text = str(text)
    # ใช้ newmm ซึ่งเสถียรกับภาษาไทย
    return word_tokenize(text, engine="newmm")
# --- IGNORE ---