import streamlit as st
import cv2
import numpy as np
from easyocr import Reader
from groq import Groq
from dotenv import load_dotenv
import os

# Initialize EasyOCR reader
easy_ocr = Reader(['en'])

# Load environment variables from .env file
load_dotenv()

# Get the API key from the .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# Function to preprocess image
def preprocess_image(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    return img

# Function to extract text using EasyOCR
def extract_text_easyocr(image):
    # Convert image to grayscale for better OCR performance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = easy_ocr.readtext(gray)
    extracted_text = " ".join([result[1] for result in results])
    return extracted_text

# Function to generate structured output using Groq
def generate_structured_output(text):
    template = f"""
    Extract the following details from the provided text:
    1. Company Name
    2. Designation
    3. Phone Number
    4. Address
    5. Email ID
    6. Web links

    Text:
    {text}
    
    Note: Don't give extra lines in the reponse only give the details asked above.
    """
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": template}],
        model="llama3-8b-8192"
    )
    answer = chat_completion.choices[0].message.content
    return answer

# Streamlit UI
st.title("Business Card Text Extraction-2")

# Camera input
image_input = st.camera_input("Capture a business card")

if image_input:
    # Preprocess image
    image = preprocess_image(image_input.read())
    
    # Process when the user clicks "Submit"
    if st.button("Submit"):
        # Extract text using EasyOCR
        extracted_text = extract_text_easyocr(image)
    
    # Display structured output
    if 'extracted_text' in locals():
        structured_output = generate_structured_output(extracted_text)
        st.subheader("Available Details in Business card")
        st.text(structured_output)

