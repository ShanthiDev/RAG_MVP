# streamlit_app/pages/01_Prompts.py
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from src.rag.answer import SYSTEM_PROMPT, USER_GUIDE

st.set_page_config(page_title="Prompts — What do they say?", layout="wide")
st.title("✍️ Edit prompts (session only)")

# Initialize session state with defaults once
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = SYSTEM_PROMPT
if "user_guide" not in st.session_state:
    st.session_state.user_guide = USER_GUIDE

with st.form("prompt_form", clear_on_submit=False):
    sys_p = st.text_area("System prompt", st.session_state.system_prompt, height=260)
    usr_p = st.text_area("User guide / Output format", st.session_state.user_guide, height=260)
    col1, col2, col3 = st.columns([1,1,3])
    with col1:
        save = st.form_submit_button("Save")
    with col2:
        reset = st.form_submit_button("Reset to defaults")

if save:
    st.session_state.system_prompt = sys_p
    st.session_state.user_guide = usr_p
    st.success("Saved for this browser session.")

if reset:
    st.session_state.system_prompt = SYSTEM_PROMPT
    st.session_state.user_guide    = USER_GUIDE
    st.info("Reset to project defaults.")
