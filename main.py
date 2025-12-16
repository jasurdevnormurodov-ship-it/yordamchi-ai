# from types import None
import httpx
import streamlit as st
from mistralai import Mistral
import time
import re
import pandas as pd
import base64
import yordamchi_mistral_ai

# ==================== PAGE CONFIG (HAR DOIM ENG YUQORIDA) ====================
st.set_page_config(
    page_title="Yordamchi AI",
    layout="wide",                     # 🔥 WIDE MODE DOIM YOQIQ
    initial_sidebar_state="collapsed",
    page_icon='artbreeder-poser.ico'
)

# ==================== STREAMLIT UI NI YASHIRISH ====================
st.markdown("""
<style>
header {visibility: hidden;}              /* Fork / Deploy */
footer {visibility: hidden;}               /* Streamlit footer */
[data-testid="stToolbar"] {visibility: hidden;}  /* Past o‘ng ikon */
</style>
""", unsafe_allow_html=True)

# ==================== BACKGROUND IMAGE ====================
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

# ==================== SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "Your name is Yordamchi AI. You were created by SAITUG."
        }
    ]

if "is_pro" not in st.session_state:
    st.session_state.is_pro = False

# ==================== JAVOB FORMATLASH ====================
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

# ==================== TYPING ANIMATION ====================
def type_writer(text, delay=0.02):
    placeholder = st.empty()
    current_text = ""

    for char in text:
        current_text += char
        placeholder.markdown(current_text + "▌")
        time.sleep(delay)

    placeholder.markdown(current_text)


# ==================== PRO MODAL ====================
@st.dialog("⚠️ Cheklovga yetdingiz")
def pro_dialog():
    st.warning("Siz 5 martadan ko‘p savol berdingiz.")
    st.markdown("""
    **Takliflar:**
    - 🆕 Yangi chat ochish
    - ✂️ Savolni qisqartirish
    - ⭐ PRO rejimga o‘tish (cheksiz)
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔓 PRO ni faollashtirish"):
            st.session_state.is_pro = True
            st.success("PRO faollashtirildi!")
            time.sleep(1)
            st.rerun()

    with col2:
        if st.button("❌ Yopish"):
            st.rerun()

# ==================== MAIN APP ====================
try:
    api_key = "VwcSrQywHXQ1Gjh4htEG5LgwqvpzJdJc"
    model = "mistral-medium-latest"
    client = Mistral(api_key=api_key)

    st.title("💬 Yordamchi AI")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "Your name is Yordamchi AI, you were created by SAITUG (Scienstic and informational technologyof university-group,  Ilm-fan va Axborot Texnologiyalariga yo'naltirilgan universitet-guruh)"}
        ]

    for msg in st.session_state.messages[1:]:
        with st.chat_message(msg["role"]):
            render_reply(msg["content"])

    user_input = st.chat_input("Savolingizni yozing...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        if user_input:
            chat_count = len(st.session_state.messages) - 1

        # ❌ LIMIT
        if chat_count >= 5 and not st.session_state.is_pro:
            pro_dialog()
            st.stop()

        # USER MESSAGE
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # MISTRAL API
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model="mistral-medium-latest",
            messages=st.session_state.messages
        )

        reply = response.choices[0].message.content
        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )

        # ASSISTANT (typing animation)
        with st.chat_message("assistant"):
            type_writer(reply)

except httpx.ConnectError:
    st.error("Sizda tarmog' faolligi mavjud emas, Iltimos, Internetga, WiFiga yoki boshqa tarmoqqa ulaning!")
except Exception as e:
    st.error(f"Xatolik: {e}")
# except Nonetype as N:
    st.error(f"Iltioms, savollaringizni javobini Google, Wikipedia yoki shu kabi saytlar yoki dasturlardan tekshiring, xato javob berish ehtimoli mavjud !")


