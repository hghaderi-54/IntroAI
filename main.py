import os
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in .env")

# Initialize OpenAI client
def create_client() -> OpenAI:
    return OpenAI(api_key=api_key)

# Read PDF and convert to text
def pdf_to_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

# Prepare messages for OpenAI
SYSTEM_PROMPT = """You are an assistant that translates text from English to Farsi (Persian).
Respond only with the translated text. Do not include any explanations or additional commentary.
"""
USER_PROMPT_PREFIX = "Please translate the following text from English to Farsi:"

def create_messages(pdf_text: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": f"{USER_PROMPT_PREFIX.strip()}\n\n{pdf_text}"}
    ]

# Create OpenAI response
def create_response(client: OpenAI, messages: list) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )
    return response.choices[0].message.content

# Main function
def main(pdf_file: str):
    client = create_client()
    pdf_text = pdf_to_text(pdf_file)
    messages = create_messages(pdf_text)
    translation = create_response(client, messages)
    
    print("----- PDF TRANSLATION -----")
    print(translation)

# Entry point
if __name__ == "__main__":
    pdf_file = "story.pdf"  # Replace with your PDF file path
    main(pdf_file)
