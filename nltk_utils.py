import numpy as np
from pythainlp.tokenize import word_tokenize  # ใช้สำหรับแยกคำภาษาไทย

def tokenize(sentence):
    """
    แยกคำในประโยคให้ออกมาเป็นรายการของคำ (tokens)
    """
    return word_tokenize(sentence, keep_whitespace=False)  # แยกคำไทยโดยไม่เก็บช่องว่าง

def bag_of_words(tokenized_sentence, words):
    """
    สร้าง Bag of Words:
    คืนค่าอาร์เรย์ที่มีค่า 1 หากคำใน `words` ปรากฏใน `tokenized_sentence`
    
    ตัวอย่าง:
    sentence = ["สวัสดี", "คุณ", "เป็น", "อย่างไร"]
    words = ["สวัสดี", "ลาก่อน", "ขอบคุณ", "คุณ"]
    bog   = [   1   ,    0   ,     0   ,   1  ]
    """
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1
    return bag
