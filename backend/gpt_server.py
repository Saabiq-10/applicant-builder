from flask import Flask, request, jsonify
from flask_cors import CORS
import json, os, re, time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

app = Flask(__name__)
CORS(app)

here = os.path.dirname(__file__)
model = SentenceTransformer("all-MiniLM-L6-v2")

with open(os.path.join(here, "teams_with_embeddings.json"), "r", encoding="utf-8") as f:
    opportunities = json.load(f)

def compute_score(subteam, job_embedding, required_skills, required_tools):
    subteam_embedding = np.array(subteam["embedding"]).reshape(1, -1)
    sim = cosine_similarity(job_embedding, subteam_embedding)[0][0]
    skills = set(subteam.get("skills", []))
    tools = set(subteam.get("tools", []))
    skill_overlap = len(skills & required_skills) / max(len(required_skills), 1)
    tool_match = len(tools & required_tools) / max(len(required_tools), 1)
    return 0.5 * sim + 0.3 * skill_overlap + 0.2 * tool_match

def filter_courses(job_desc, all_courses):
    jd_lower = job_desc.lower()
    ai_keywords = ["artificial intelligence", "ai", "machine learning", "ml", "deep learning"]
    python_keywords = ["python", "programming", "developer"]
    is_ai = any(k in jd_lower for k in ai_keywords)
    is_python = any(k in jd_lower for k in python_keywords)
    return [c for c in all_courses if
            ("ai" in c.get("tags", []) and is_ai) or
            ("python" in c.get("tags", []) and is_python)]

def summarize_recommendations(student_teams, hackathons, courses):
    def short(t): return f"{t['name']} - {t.get('url', 'no link')}"
    return {
        "student_teams": [short(t) for t in student_teams],
        "hackathons": [short(h) for h in hackathons[:3]],
        "courses": [short(c) for c in courses[:3]]
    }

def build_prompt(job_desc, summary):
    return f"""You are an assistant recommending opportunities based on the following job description:

Job Description:
{job_desc}

Top Matching Student Clubs:
{summary['student_teams']}

Top Hackathons:
{summary['hackathons']}

Top Courses:
{summary['courses']}

Return ONLY a valid JSON object with exactly these keys: "student_teams", "hackathons", and "courses".

Each key must map to a list of objects with:
- "name": string
- "reason": 1-line reason it's relevant (string)
- "url": a direct clickable link (string)

Example:

{{
  "student_teams": [
    {{
      "name": "MetRocketry",
      "reason": "Rocket design team that builds and launches rockets.",
      "url": "https://example.com"
    }}
  ],
  "hackathons": [
    {{
      "name": "Lunaris Hacks",
      "reason": "Large student-led hackathon in Canada.",
      "url": "https://example.com"
    }}
  ],
  "courses": [
    {{
      "name": "Intro to AI with Python",
      "reason": "Harvard course teaching AI basics.",
      "url": "https://example.com"
    }}
  ]
}}

DO NOT include code blocks, text before or after, or comments. Your response MUST be a clean JSON object that starts with {{ and ends with }}.
"""

@app.route("/generate", methods=["POST"])
def generate():
    jd = request.get_json(force=True).get("prompt", "").strip()
    if not jd:
        return jsonify({"error": "No job description provided"}), 400

    job_vec = model.encode(jd).reshape(1, -1)
    required_skills = {w for w in jd.lower().split() if w in {"robotics", "design", "integration", "autonomous"}}
    required_tools = {w for w in jd.lower().split() if w in {"python", "cad", "ros", "c++", "solidworks"}}

    top_teams = []
    for team in opportunities["student_teams"]:
        if not team.get("subteams"):
            continue
        scored = []
        for st in team["subteams"]:
            if not isinstance(st, dict) or "embedding" not in st:
                continue
            scored.append((compute_score(st, job_vec, required_skills, required_tools), st))
        if not scored:
            continue
        scored.sort(reverse=True)
        best_subs = [st for _, st in scored[:2]]
        team_copy = {
            "name": team["name"],
            "url": team.get("url", ""),
            "subteams": best_subs
        }
        top_teams.append((scored[0][0], team_copy))

    top_teams.sort(reverse=True)
    top_teams = [t for _, t in top_teams[:3]]
    filtered_courses = filter_courses(jd, opportunities["courses"])
    top_hackathons = opportunities["hackathons"][:3]
    summary = summarize_recommendations(top_teams, top_hackathons, filtered_courses)
    prompt = build_prompt(jd, summary)

    try:
        print("⏳ Sending prompt to LM Studio server...")
        start_time = time.time()

        response = requests.post(
            "http://127.0.0.1:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "WizardLM-2-7B",
                "messages": [
                    {"role": "system", "content": "You are an assistant recommending opportunities based on job descriptions."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
                "temperature": 0.7
            },
            timeout=60
        )

        duration = time.time() - start_time
        print(f"✅ Model responded in {duration:.2f} seconds")

        raw_output = response.json()["choices"][0]["message"]["content"]
        print("🔁 Raw model output:", raw_output)

        # Strip code fences
        raw_output = re.sub(r"^```(?:json)?|```$", "", raw_output.strip(), flags=re.MULTILINE).strip()

        # Fix known formatting issues
        raw_output = re.sub(r'\](\s*)"(courses|hackathons|student_teams)"', r'], "\2"', raw_output)
        raw_output = re.sub(r'}\s*{', r'}, {', raw_output)  # fix missing commas between objects

        # Extract and parse valid JSON
        match = re.search(r'{.*}', raw_output, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model output.")

        json_block = match.group()

        # Balance braces if needed
        if json_block.count('{') > json_block.count('}'):
            json_block += '}'

        parsed = json.loads(json_block)
        return jsonify(parsed)

    except Exception as e:
        print(f"❌ Error during model call or JSON parsing: {e}")
        return jsonify({
            "error": "Model failed or returned invalid JSON.",
            "debug": str(e),
            "raw_output": raw_output
        }), 200

if __name__ == "__main__":
    app.run(debug=True, port=3000)
