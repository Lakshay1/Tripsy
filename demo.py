import streamlit as st
import os
import shutil
from orchestrator import AnthropicOrchestrator
from orchestrator import (
   EMAIL_TOOL_DEF,
   UNDERSTAND_IMAGE_TOOL_DEF,
   UNDERSTAND_VIDEO_TOOL_DEF,
   fetch_emails,
   get_image_location,
   get_video_location,
)


BASE_PROMPT = '/Users/lakshayk/Developer/Tripsy/Tripsy/base_prompt.txt'

# Set up orchestrator once
orchestrator = AnthropicOrchestrator(api_key="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")
orchestrator.register_tool(EMAIL_TOOL_DEF, fetch_emails)
orchestrator.register_tool(UNDERSTAND_IMAGE_TOOL_DEF, get_image_location)
orchestrator.register_tool(UNDERSTAND_VIDEO_TOOL_DEF, get_video_location)

# Streamlit App UI
st.set_page_config(page_title="🧳 Tripsy", page_icon="🧳")
st.title("🧳 Tripsy - Your AI Trip Assistant")
st.write(
   "Welcome to Tripsy! I am your personal travel organizer. You can ask me to fetch your trip-related emails, organize it one place and make suggestions for your trip."
)

user_input_text = st.text_input("Enter your request here:")
uploaded_file = st.file_uploader("Upload an image or video (optional)", type=["jpg", "jpeg", "png", "mp4"])
run_button = st.button("Generate")

base_prompt = ""
try:
    with open(BASE_PROMPT, 'r') as file:
        # Read the entire content of the file
        base_prompt = file.read()
except FileNotFoundError:
    print("File not found. Please check the file path.")

if run_button:
    with st.spinner("Fetching your trip details..."):
        if uploaded_file is not None:
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            default_input = f" I have uploaded a video at {file_path}."
            user_input_text += default_input

        prompt = base_prompt.replace("{{user_prompt}}", user_input_text)
        result = orchestrator.chat(prompt)
        st.success(result.content[0].text)
