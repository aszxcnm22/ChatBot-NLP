import streamlit as st
import torch
import numpy as np
import random
import json
import asyncio
from model import NeuralNet
from nltk_utils import bag_of_words
from pythainlp.tokenize import word_tokenize

# ตรวจสอบและแก้ไขปัญหา asyncio event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

# โหลดข้อมูล intents.json
with open("intents.json", "r", encoding="utf-8") as json_data:
    intents = json.load(json_data)

# โหลดโมเดลที่เทรนไว้
FILE = "data.pth"
data = torch.load(FILE, map_location=torch.device('cpu'))

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# โหลดโมเดลเข้า Streamlit
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# ตั้งค่าหน้าตา Streamlit
st.set_page_config(page_title="แชทบอท AI", page_icon="🤖", layout="centered")

st.title("🤖 แชทบอทภาษาไทย AI")
st.markdown("พิมพ์ข้อความด้านล่างเพื่อสนทนากับบอท!")

# ใช้ session state เพื่อเก็บประวัติการสนทนา
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

bot_name = "Sam"

# ฟังก์ชันตอบกลับจากบอท
def get_response(sentence):
    sentence_words = word_tokenize(sentence, keep_whitespace=False)
    X = bag_of_words(sentence_words, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.50:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "ขอโทษค่ะ/ครับ ฉันไม่เข้าใจคำถามของคุณ 😕 กรุณาลองใหม่"

# ฟังก์ชันส่งข้อความ
def submit_chat():
    user_text = st.session_state.input.strip()
    if user_text:
        response = get_response(user_text)
        st.session_state.chat_history.append(("คุณ", user_text))
        st.session_state.chat_history.append(("🤖 Sam", response))
        st.session_state.input = ""  # ล้างช่องข้อความ

st.session_state.submit_chat = submit_chat

# กล่องป้อนข้อความผู้ใช้
user_input = st.text_input("📝 พิมพ์ข้อความของคุณที่นี่: ", key="input", on_change=submit_chat)

# แสดงประวัติการสนทนา
st.markdown("### 📜 ประวัติการสนทนา")
for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "คุณ" else "assistant"):
        st.write(f"**{sender}:** {message}")

# ปุ่มล้างประวัติการสนทนา
if st.button("🗑️ ล้างประวัติแชท"):
    st.session_state.chat_history = []

# ตัวอย่างคำถาม
st.markdown("---")
st.subheader("📌 ตัวอย่างคำถามที่คุณสามารถลองพิมพ์")
with st.expander("📖 คำถามที่สามารถถามได้"):
    for intent in intents['intents']:
        st.write(f"- {', '.join(intent['patterns'])}")

st.markdown("""
🚀 **Tip:** ลองถามคำถามที่หลากหลายเพื่อดูว่าบอทสามารถตอบอะไรได้บ้าง!
""")
