import streamlit as st
from paddleocr import PaddleOCR
from langchain_community.llms import Ollama
import json

# Initialize the OCR model for English
def Initialize_ocr(lang="en"):
    ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    if ocr is None:
        raise ValueError("Failed to initialize the OCR model.")
    return ocr

# Extract text from image
def Perform_ocr(image_path):
    ocr_model = Initialize_ocr()
    image_text = ocr_model.ocr(image_path, cls=True)
    if image_text is None:
        raise ValueError("Failed to extract text")
    return image_text

# Extract plain text from OCR result
def extract_plain_text(image_text):
    plain_text = ""
    for line in image_text:
        for entry in line:
            text = entry[1][0]
            plain_text += text + "\n"
    return plain_text.strip()

# Initialize the LLM
def Initialize_llm():
    llm = Ollama(model="llama3", temperature=1, format='json')
    return llm

# Custom prompt template
prompt_template = """

You can as a text summarization specialist. Extract key information from this text in the form of key-value pairs.

Text:
{plain_text}
"""

def call_ocr_model(image_path):
    image_text = Perform_ocr(image_path)
    texts = extract_plain_text(image_text)
    return texts

st.header("Invoices Data Extraction BOT")

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Perform OCR on the uploaded image
    try:
        plain_text = call_ocr_model("temp_image.png")
        st.text_area("Extracted Text", plain_text, height=300)

        # Generate prompt for LLM
        prompt = prompt_template.format(plain_text=plain_text)

        # Call the LLM to get JSON output
        llm = Initialize_llm()
        response = llm(prompt)

        # Display the response
        st.json(response)

    except Exception as e:
        st.error(f"Error: {e}")
