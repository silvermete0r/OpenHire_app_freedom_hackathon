import os
import uuid
import shutil
import sqlite3
import re
from pathlib import Path
from docx2pdf import convert as docx_to_pdf
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path("./")
RESUME_DIR = BASE_DIR / "data/test_data"
TRASH_DIR = BASE_DIR / "data/trash_data"
FILTERED_DIR = BASE_DIR / "data/filtered_data"
DB_PATH = BASE_DIR / "data/resumes.db"

SUPPORTED_FORMATS = {'.pdf', '.docx'}

def initialize_directories():
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)

def convert_doc_to_pdf(doc_file, pdf_path):
    if doc_file.suffix == '.docx':
        docx_to_pdf(doc_file, pdf_path)

def extract_email(text):
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return email_match.group(0) if email_match else "order@ftel.kz"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def extract_features_and_insert_into_db():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id TEXT PRIMARY KEY,
            email TEXT,
            resume_text TEXT,
            tfidf_features TEXT
        )
    ''')

    tfidf_vectorizer = TfidfVectorizer()

    for resume_file in FILTERED_DIR.iterdir():
        text = extract_text_from_pdf(resume_file)

        email = extract_email(text)

        tfidf_matrix = tfidf_vectorizer.fit_transform([text])
        tfidf_features = tfidf_matrix.toarray().tolist()[0]

        resume_id = resume_file.stem 

        cursor.execute('''
            INSERT OR REPLACE INTO resumes (id, email, resume_text, tfidf_features)
            VALUES (?, ?, ?, ?)
        ''', (resume_id, email, text, str(tfidf_features)))

    connection.commit()
    connection.close()

def filter_resumes():
    for resume_file in RESUME_DIR.iterdir():
        file_ext = resume_file.suffix.lower()

        if file_ext not in SUPPORTED_FORMATS:
            new_name = f"{uuid.uuid4()}{file_ext}"
            shutil.move(resume_file, TRASH_DIR / new_name)
            continue

        if file_ext == '.docx':
            new_name = f"{uuid.uuid4()}.pdf"
            pdf_path = FILTERED_DIR / new_name
            convert_doc_to_pdf(resume_file, pdf_path)

        elif file_ext == '.pdf':
            new_name = f"{uuid.uuid4()}.pdf"
            shutil.move(resume_file, FILTERED_DIR / new_name)

        resume_file.unlink(missing_ok=True)

initialize_directories()
filter_resumes()
extract_features_and_insert_into_db()