import streamlit as st
import uuid
import os
from datetime import datetime
from pathlib import Path
from pdf_ingest import PdfIngestor  # your wrapper
from csv_ingest import CsvIngestor      # your class
from views import run_assistant_workflow # LangGraph assistant entrypoint

# --- Setup ---
st.set_page_config(page_title="Chat Assistant", layout="wide")

# 🔧 Add your custom styling here
st.markdown("""
<style>
.user-bubble {
    background-color:#DCF8C6;
    color: black;
    padding:10px;
    border-radius:10px;
    margin-bottom:5px;
    width:fit-content;
}
.bot-bubble {
    background-color:#F1F0F0;
    color: black;
    padding:10px;
    border-radius:10px;
    margin-bottom:5px;
    width:fit-content;
}
.chat-box {
    height:calc(100vh - 200px);
    overflow-y:auto;
    padding-right:20px;
}
.stTextInput>div>div>input {
    font-size:16px;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = "uploaded_files"

# --- Session State ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

if "current_session" not in st.session_state:
    new_session = str(uuid.uuid4())
    st.session_state.current_session = new_session
    st.session_state.chat_sessions[new_session] = []

# Adding flag to track whether the file has been ingested
if "file_ingested" not in st.session_state:
    st.session_state.file_ingested = False  # File hasn't been ingested initially

# Store parsed_file_path
if "parsed_file_path" not in st.session_state:
    st.session_state.parsed_file_path = None

user_id = "user123"  # static or dynamic per login
session_id = st.session_state.current_session
session_folder = Path(BASE_DIR) / user_id / session_id
session_folder.mkdir(parents=True, exist_ok=True)

# --- Sidebar: Upload & Chat History ---
st.sidebar.title("🧠 Chat Assistant")
st.sidebar.markdown("### 💬 Chat Sessions")

for sid in st.session_state.chat_sessions.keys():
    if st.sidebar.button(f"Chat {sid[:6]}", key=sid):
        st.session_state.current_session = sid
        st.session_state.file_ingested = False  # Reset on session switch
        st.session_state.parsed_file_path = None

if st.sidebar.button("🆕 New Chat"):
    new_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_id] = []
    st.session_state.current_session = new_id
    st.session_state.file_ingested = False  # Reset file ingestion flag
    st.session_state.parsed_file_path = None
    st.rerun()

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📎 Upload File", type=["pdf", "csv"], label_visibility="collapsed")

# --- File Ingestion ---
file_path = None
if uploaded_file and not st.session_state.file_ingested:
    file_path = session_folder / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Ingest the file if it's a PDF or CSV
    if file_path.suffix == ".pdf":
        parsed_file_path = PdfIngestor(str(file_path), user_id, session_id).ingest()
        if not parsed_file_path:
            st.sidebar.error("Ingestion failed: parsed_file_path is None")
            st.session_state.file_ingested = False
        elif not os.path.exists(parsed_file_path):
            st.sidebar.error(f"Parsed file not found at {parsed_file_path}")
            st.session_state.file_ingested = False
        else:
            st.session_state.parsed_file_path = parsed_file_path
            st.session_state.file_ingested = True
            st.sidebar.success(f"Ingested PDF: {uploaded_file.name}")
    elif file_path.suffix == ".csv":
        csv_ingestor = CsvIngestor()
        csv_ingestor.ingest_csv_files([uploaded_file], user_id, session_id)
        st.session_state.parsed_file_path = str(file_path)  # For CSV, use original file path
        st.session_state.file_ingested = True
        st.sidebar.success(f"Ingested CSV: {uploaded_file.name}")

# --- Chat Area Styling ---
st.markdown("""
<style>
.user-bubble {background-color:#DCF8C6;padding:10px;border-radius:10px;margin-bottom:5px;width:fit-content;}
.bot-bubble {background-color:#F1F0F0;padding:10px;border-radius:10px;margin-bottom:5px;width:fit-content;}
.chat-box {height:calc(100vh - 200px);overflow-y:auto;padding-right:20px;}
.stTextInput>div>div>input {font-size:16px;}
</style>
""", unsafe_allow_html=True)

# --- Chat Display ---
st.title("🤖 RAG Application")
chat_area = st.container()
chat_area.markdown("<div class='chat-box'>", unsafe_allow_html=True)

history = st.session_state.chat_sessions[session_id]
if not history:
    st.markdown("<h3 style='color: #1f77b4;'>Hello, ask me anything!</h3>", unsafe_allow_html=True)

for msg in history:
    role = msg["role"]
    bubble = "user-bubble" if role == "user" else "bot-bubble"
    chat_area.markdown(f"<div class='{bubble}'>{msg['text']}</div>", unsafe_allow_html=True)

chat_area.markdown("</div>", unsafe_allow_html=True)

# --- Chat Input ---
query = st.chat_input("Type your message here...")

# --- Handle Query ---
if query:
    history.append({"role": "user", "text": query})
    
    if not st.session_state.file_ingested or not st.session_state.parsed_file_path:
        history.append({"role": "assistant", "text": "❗ Please upload a file and ensure it is ingested."})
    else:
        try:
            print(st.session_state.parsed_file_path)
            result = run_assistant_workflow(user_id, session_id, st.session_state.parsed_file_path, query)
            final_msgs = result.get("messages") or result.get("finalize", {}).get("messages", [])
            
            if final_msgs:
                response = final_msgs[-1].content
            else:
                response = "❌ No response generated."
            
            history.append({"role": "assistant", "text": response})
        except Exception as e:
            history.append({"role": "assistant", "text": f"❌ Error: {str(e)}"})
    
    st.rerun()