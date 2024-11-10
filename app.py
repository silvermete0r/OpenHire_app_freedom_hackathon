from flask import Flask, render_template, request, send_from_directory
import csv
from utils.tf_idf_ranking import get_top_k_candidates
from utils.ai_funs import get_ai_feedback_for_resume
import json
import os

app = Flask(__name__)

CSV_PATH = 'data/top_k_ranked_resumes.csv'

headers = {
    'Content-Type': 'application/json',
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/get_info_by_id', methods=['GET'])
def get_info_by_id():
    id = request.args.get('id', '')
    with open(CSV_PATH, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['id'] == id:
                return json.dumps(row, ensure_ascii=False)
    return json.dumps({}, ensure_ascii=False)

@app.route('/api/get_positions', methods=['GET'])
def get_positions():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'positions.json')

@app.route('/api/get_ai_analysis', methods=['GET'])
def get_ai_analysis():
    candidate_id = request.args.get('candidate_id', '')
    job_requirements = request.args.get('requirements', '')
    candidate_info = json.loads(get_info_by_id(candidate_id))
    resume_text = candidate_info.get('resume_text', '')
    lang = candidate_info.get('lang', 'en')
    ai_feedback = get_ai_feedback_for_resume(resume_text=resume_text, requirements_text=job_requirements, lang=lang)
    return json.dumps(ai_feedback, ensure_ascii=False)

@app.route('/rank_candidates', methods=['GET', 'POST'])
def rank_candidates():
    if request.method == 'POST':
        position = request.form.get('position', '')
        requirements = request.form.get('requirements', '')
        k = int(request.form.get('k', 5))
        query = position + '\n' + requirements

        get_top_k_candidates(query, k)

        ranked_resumes = []
        with open(CSV_PATH, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            ranked_resumes = list(reader)

        return render_template('search_results.html', resumes=ranked_resumes, query=query)

    return render_template('rank_candidates.html')

if __name__ == '__main__':
    app.run(debug=True)