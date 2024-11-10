import sqlite3
import csv
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

DB_PATH = "data/resumes.db"
OUTPUT_CSV_PATH = "data/top_k_ranked_resumes.csv"

def load_stopwords():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    english_stopwords = set(ENGLISH_STOP_WORDS)
    
    russian_stopwords_url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.txt"
    russian_stopwords = set(requests.get(russian_stopwords_url).text.splitlines())
    
    kazakh_stopwords_url = "https://raw.githubusercontent.com/silvermete0r/QazNLTK/master/special_words/stop_words.txt"
    kazakh_stopwords = set(requests.get(kazakh_stopwords_url).text.splitlines())
    
    return english_stopwords, russian_stopwords, kazakh_stopwords

def detect_language(text):
    if any(char in 'а-яА-Я' for char in text):
        return 'ru'  # Russian
    elif any(char in 'ққғүұіһә' for char in text):
        return 'kk'  # Kazakh
    else:
        return 'en'  # Default to English

def preprocess_text(text, language, english_stopwords, russian_stopwords, kazakh_stopwords):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()

    if language == 'en':
        stop_words = english_stopwords
    elif language == 'ru':
        stop_words = russian_stopwords
    else:
        stop_words = kazakh_stopwords
    
    filtered_words = [word for word in words if word.lower() not in stop_words]

    return " ".join(filtered_words)

# Get resumes from the database
def get_resumes_from_db():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute('SELECT id, email, resume_text FROM resumes')
    resumes = cursor.fetchall()
    connection.close()
    return resumes

# Rank resumes by query and return top k candidates
def rank_resumes_by_query(query, k=None):
    resumes = get_resumes_from_db()
    
    english_stopwords, russian_stopwords, kazakh_stopwords = load_stopwords()

    query_language = detect_language(query)
    processed_query = preprocess_text(query, query_language, english_stopwords, russian_stopwords, kazakh_stopwords)
    
    tfidf_vectorizer = TfidfVectorizer(stop_words=None) 

    resume_texts = [preprocess_text(resume[2], detect_language(resume[2]), english_stopwords, russian_stopwords, kazakh_stopwords) for resume in resumes]
    
    all_texts = [processed_query] + resume_texts 

    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

    query_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    similarity_scores = cosine_similarity(query_vector, resume_vectors).flatten()

    ranked_resumes = []
    for i, (resume_id, email, text) in enumerate(resumes):
        ranked_resumes.append({
            'id': resume_id,
            'email': email,
            'resume_text': text,
            'score': similarity_scores[i],
            'lang': query_language
        })

    ranked_resumes = sorted(ranked_resumes, key=lambda x: x['score'], reverse=True)

    if k:
        ranked_resumes = ranked_resumes[:k]

    return ranked_resumes

def save_ranked_resumes_to_csv(ranked_resumes):
    with open(OUTPUT_CSV_PATH, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['id', 'email', 'resume_text', 'score', 'lang'])
        writer.writeheader()
        for resume in ranked_resumes:
            writer.writerow(resume)

def get_top_k_candidates(query, k=5):
    ranked_resumes = rank_resumes_by_query(query, k)
    save_ranked_resumes_to_csv(ranked_resumes)

if __name__ == '__main__':
    query = "Software engineer with experience in machine learning and Python"
    get_top_k_candidates(query, k=5)
