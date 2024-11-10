import json
import requests

url = "http://localhost:11434/api/generate"

headers = {
    'Content-Type': 'application/json',
}

prompt_template = '''
Analyze the following resume and provide score 0-100 and main points on the candidate's suitability for the given requirements.

Job Requirements:
{requirements_text} 

Candidate's Resume:
{resume_text}

Provide analysis in JSON format without any additional text and symbols, as a row text! Response content language should be {lang}
Example:
{
    "score": 80,
    "main_points": [
        "The candidate has experience with Python and SQL.",
        "The candidate has experience with data analysis."
    ]
}
'''

def get_ai_feedback_for_resume(resume_text, requirements_text, lang):
    full_prompt = prompt_template.format(resume_text=resume_text, requirements_text=requirements_text, lang=lang)

    data = {
        "model": "gemma2:2b",
        "stream": False,
        "prompt": full_prompt,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        actual_response = data["response"]
        return actual_response
    else:
        print("Error:", response.status_code, response.text)
        return None