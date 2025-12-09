from types import NoneType

import httpx
import streamlit as st
from mistralai import Mistral
import time
import re
import pandas as pd

import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# 🔧 Javobni formatlab chiqaruvchi funksiya
def render_reply(reply: str):
    lines = reply.splitlines()

    inside_code = False
    code_lang = None
    code_buf = []

    inside_math = False
    math_delim = None
    math_buf = []

    inside_table = False
    table_buf = []

    def flush_table():
        nonlocal table_buf
        if not table_buf:
            return
        rows = [re.split(r"\s*\|\s*", row.strip("| ")) for row in table_buf if row.strip()]
        if len(rows) >= 2:
            header = rows[0]
            # Ikkinchi qatorda chiziqlar bo‘lsa uni tashlab ketamiz
            data = rows[2:] if all(re.fullmatch(r"-+", c) for c in rows[1]) else rows[1:]
            df = pd.DataFrame(data, columns=header)
            st.table(df)
        table_buf = []

    def render_line_with_inline_math(s: str):
        # Rasm
        img_match = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', s)
        if img_match and s.strip().startswith("!"):
            for url in img_match:
                st.image(url)
            return

        # Video
        v = re.match(r'^\[(?i:video)\]\(([^)]+)\)', s)
        if v:
            st.video(v.group(1))
            return

        # YouTube havolalarini video sifatida
        yt = re.search(r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[^\s]+)', s)
        if yt:
            st.video(yt.group(1))
            return

        # Inline math ($...$)
        parts = re.split(r'(\$[^$]+\$)', s)
        for part in parts:
            if not part:
                continue
            if part.startswith("$") and part.endswith("$"):
                st.latex(part[1:-1])
            else:
                st.markdown(part)

    for raw in lines:
        line = raw.rstrip()

        # --- Kod bloklari ---
        if line.strip().startswith("```"):
            flush_table()
            if not inside_code:
                inside_code = True
                header = line.strip().strip("`")
                code_lang = header.replace("```", "").strip() or None
                code_buf = []
            else:
                inside_code = False
                content = "\n".join(code_buf)
                if (code_lang or "").lower() in {"latex", "tex", "math"}:
                    st.latex(content)
                else:
                    st.code(content, language=code_lang)
                code_lang = None
                code_buf = []
            continue

        if inside_code:
            code_buf.append(raw)
            continue

        # --- Math bloklari ---
        s = line.strip()
        if s.startswith("$$") and s.endswith("$$"):
            flush_table()
            st.latex(s[2:-2].strip())
            continue
        if s.startswith(r"\[") and s.endswith(r"\]"):
            flush_table()
            st.latex(s[2:-2].strip())
            continue

        # --- Jadval ---
        if s.startswith("|") and s.endswith("|"):
            table_buf.append(s)
            continue
        else:
            if table_buf:
                flush_table()

        # Oddiy satr
        if s:
            render_line_with_inline_math(raw)
        else:
            st.write("")

    # Oxirgi jadvalni chiqarish
    if table_buf:
        flush_table()


try:
    # 🔑 API ma'lumotlari
    api_key = "VwcSrQywHXQ1Gjh4htEG5LgwqvpzJdJc"
    model = "mistral-medium-latest"

    # 🔌 Mistral API bilan bog'lanish
    client = Mistral(api_key=api_key)

    # 📋 Streamlit sozlamalari
    st.set_page_config(page_title="Yordamchi AI", layout="centered", page_icon='artbreeder-poser.ico')
    st.title("💬 Yordamchi AI")

    # 💾 Chat tarixini saqlash
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "Your name is Yordamchi AI, you were created by Normurodov Jasur Ibragimovich"}
        ]

    # 📜 Oldingi xabarlarni chiqarish
    for msg in st.session_state.messages[1:]:  # system xabarni tashlab
        with st.chat_message(msg["role"]):
            render_reply(msg["content"])

    # 💬 Foydalanuvchi inputi
    user_input = st.chat_input("Savolingizni yozing...")

    if user_input:
        # ➕ Foydalanuvchi xabarini qo'shish
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # 🤖 AI javobini olish
        response = client.chat.complete(
            model=model,
            messages=st.session_state.messages
        )
        reply = response.choices[0].message.content

        # ➕ Javobni qo'shish
        st.session_state.messages.append({"role": "assistant", "content": reply})

        # ⌛️ Formatlangan javobni chiqarish
        with st.chat_message("assistant"):
            blocks = render_reply(reply)  # oddiy matnlarni qaytaradi
            for block in blocks:
                placeholder = st.empty()
                animated_text = ""
                show_dot = True
                for char in block:
                    animated_text += char
                    dot = "⚫" if show_dot else "&nbsp;"
                    placeholder.markdown(animated_text + dot, unsafe_allow_html=True)
                    show_dot = not show_dot
                    time.sleep(0.02)
                placeholder.markdown(animated_text)

except httpx.ConnectError:
    st.write('Iltimos internetga ulaning!')
except Exception as e:
    st.error(f'Xatolik yuz berdi: {e}')
except NoneType as n:
    st.write('AI ba\'zi savollarga to\'gri javob bermaydi, iltimos javoblarni tekshiring !')
