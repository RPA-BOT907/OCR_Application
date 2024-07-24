import streamlit as st
from paddleocr import PaddleOCR
from langchain_community.llms import Ollama
import pandas as pd
import json
import tempfile
import os
import zipfile
import logging
from io import BytesIO

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Initialize the OCR model for English
def Initialize_ocr(lang="en"):
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        return ocr
    except Exception as e:
        logging.error(f"Error initializing OCR model: {e}")
        raise

# Extract text from image
def Perform_ocr(image_path):
    try:
        ocr_model = Initialize_ocr()
        image_text = ocr_model.ocr(image_path, cls=True)
        return image_text
    except Exception as e:
        logging.error(f"Error performing OCR: {e}")
        raise

# Extract plain text from OCR result
def extract_plain_text(image_text):
    plain_text = ""
    try:
        for line in image_text:
            for entry in line:
                text = entry[1][0]
                plain_text += text + "\n"
        return plain_text.strip()
    except Exception as e:
        logging.error(f"Error extracting plain text: {e}")
        raise

# Initialize the LLM
def Initialize_llm():
    try:
        llm = Ollama(model="llama3.1", temperature=1, format='json')
        return llm
    except Exception as e:
        logging.error(f"Error initializing LLM: {e}")
        raise

# Custom prompt template
prompt_template = """
You are the text summarization specialist. Extract key information from this text in the form of key-value pairs.

Text:
{plain_text}
"""

def call_ocr_model(image_path):
    image_text = Perform_ocr(image_path)
    texts = extract_plain_text(image_text)
    return texts

st.header('Invoices Data Extraction BOT:sunglasses:', divider='rainbow')

uploaded_files = st.file_uploader("Choose image files or a ZIP file", type=["png", "jpg", "jpeg", "zip"], accept_multiple_files=True)

# Initialize session state
if 'all_results' not in st.session_state:
    st.session_state['all_results'] = []
if 'files_processed' not in st.session_state:
    st.session_state['files_processed'] = False
if 'temp_image_files' not in st.session_state:
    st.session_state['temp_image_files'] = []

# Add a run button to start the data extraction process
run_button = st.button("Run")

if run_button and uploaded_files:
    st.session_state['files_processed'] = False
    st.session_state['temp_image_files'] = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name.lower().endswith('.zip'):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)

                    for file_name in os.listdir(temp_dir):
                        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            file_path = os.path.join(temp_dir, file_name)
                            st.session_state['temp_image_files'].append((file_name, file_path))

            except Exception as e:
                st.error(f"Error processing ZIP file {uploaded_file.name}: {e}")
                logging.error(f"Error processing ZIP file {uploaded_file.name}: {e}")

        else:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    st.session_state['temp_image_files'].append((uploaded_file.name, temp_file.name))

            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {e}")
                logging.error(f"Error processing file {uploaded_file.name}: {e}")

    if st.session_state['temp_image_files']:
        st.session_state['all_results'] = []

        for file_name, file_path in st.session_state['temp_image_files']:
            try:
                plain_text = call_ocr_model(file_path)
                st.text_area(f"Extracted Text from {file_name}", plain_text, height=300)

                prompt = prompt_template.format(plain_text=plain_text)
                llm = Initialize_llm()
                response = llm(prompt)
                response_json = json.loads(response)

                df = pd.DataFrame(list(response_json.items()), columns=["Key", "Value"])
                df_transposed = df.T
                df_transposed.columns = df_transposed.iloc[0]
                df_transposed = df_transposed[1:]

                st.session_state['all_results'].append(df_transposed)

            except Exception as e:
                st.error(f"Error processing file {file_name}: {e}")
                logging.error(f"Error processing file {file_name}: {e}")

        st.session_state['files_processed'] = True

if st.session_state['files_processed'] and st.session_state['all_results']:
    final_df = pd.concat(st.session_state['all_results'], keys=[file[0] for file in st.session_state['temp_image_files']], names=['File'])
    st.write("All Extracted Data")
    st.table(final_df)

    # Convert DataFrame to Excel file in memory
    excel_buffer = BytesIO()
    final_df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)

    st.download_button(
        label="Download all data as Excel",
        data=excel_buffer,
        file_name="extracted_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
