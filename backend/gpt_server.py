from flask import Flask, request, jsonify
from flask_cors import CORS
import json, os, re, logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Setup
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

here = os.path.dirname(__file__)
model = SentenceTransformer("all-MiniLM-L6-v2")

with open(os.path.join(here, "teams_with_embeddings.json"), "r", encoding="utf-8") as f:
    opportunities = json.load(f)

# Create lookup from subteam full names to URLs
opportunities_lookup = {}
for team in opportunities["student_teams"]:
    for sub in team.get("subteams", []):
        if not isinstance(sub, dict):
            logging.warning(f"Skipping malformed subteam in {team['name']}: {sub}")
            continue
        full_name = f"{team['name']} – {sub.get('name', 'Subteam')}"
        opportunities_lookup[full_name] = {"url": team.get("url", "")}


def compute_score(subteam, job_embedding, required_skills, required_tools):
    try:
        subteam_embedding = np.array(subteam["embedding"]).reshape(1, -1)
        sim = cosine_similarity(job_embedding, subteam_embedding)[0][0]
    except Exception:
        return 0.0
    skills = set(subteam.get("skills", []))
    tools = set(subteam.get("tools", []))
    skill_overlap = len(skills & required_skills) / max(len(required_skills), 1)
    tool_match = len(tools & required_tools) / max(len(required_tools), 1)
    return 0.5 * sim + 0.3 * skill_overlap + 0.2 * tool_match


def filter_courses(job_desc, all_courses):
    jd_lower = job_desc.lower()
    is_ai = any(k in jd_lower for k in ["artificial intelligence", "ai", "machine learning", "ml", "deep learning"])
    is_python = any(k in jd_lower for k in ["python", "programming", "developer"])
    return [
        c for c in all_courses if
        ("ai" in c.get("tags", []) and is_ai) or
        ("python" in c.get("tags", []) and is_python)
    ]


def truncate_list(items, max_items=5, max_chars=1000):
    lines = [f"- {item.get('name', '')}: {item.get('reason', '')}" for item in items[:max_items]]
    return '\n'.join(lines)[:max_chars]


def build_prompt(job_desc, summary):
    return f"""You are an assistant recommending opportunities based on the following job description:

Job Description:
{job_desc}

Top Matching Student Clubs:
{truncate_list(summary['student_teams'])}

Top Hackathons:
{truncate_list(summary['hackathons'])}

Top Courses:
{truncate_list(summary['courses'])}

Return ONLY a valid JSON object with exactly these keys: "student_teams", "hackathons", and "courses".

Each item must include:
- "name": string
- "reason": 1-line string
- "url": clickable link (string)

End your response after the final }}. Do not add extra text.
"""


def extract_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(f"Partial JSON parse failed: {e}")
        raise ValueError("No valid JSON found")


def get_reasons_from_llm(job_desc, teams, hackathons, courses):
    def fmt(items): return "\n".join(f"- {x['name']}" for x in items)

    prompt = f"""You are an assistant. Based on this job description:

{job_desc}

Explain briefly why each of these are relevant.

Student Teams:
{fmt(teams)}
Hackathons:
{fmt(hackathons)}
Courses:
{fmt(courses)}

Respond in this JSON format:
```json
{{
  "student_teams": {{
    "Name 1": "Reason",
    ...
  }},
  "hackathons": {{
    ...
  }},
  "courses": {{
    ...
  }}
}}
```"""

    try:
        res = requests.post(
            "http://127.0.0.1:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "wizardlm-2-7b",
                "messages": [
                    {"role": "system", "content": "You explain why each opportunity is relevant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2048
            },
            timeout=30
        )
        res.raise_for_status()
        raw = res.json()["choices"][0]["message"]["content"]
        cleaned = re.sub(r"^```json|```$", "", raw.strip())
        return extract_json(cleaned)
    except Exception as e:
        logging.error(f"LLM request failed: {e}")
        raise


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
        scored = [
            (compute_score(st, job_vec, required_skills, required_tools), st)
            for st in team.get("subteams", []) if isinstance(st, dict) and "embedding" in st
        ]
        if scored:
            scored.sort(reverse=True)
            top_teams.append((scored[0][0], {
                "name": team["name"],
                "url": team.get("url", ""),
                "subteams": [st for _, st in scored[:2]]
            }))

    top_teams.sort(reverse=True)
    top_teams = [t for _, t in top_teams[:3]]

    hackathons = opportunities["hackathons"][:3]
    courses = filter_courses(jd, opportunities["courses"])

    # Flatten subteams into individual items with full names
    team_items = [{
        "name": f"{team['name']} – {sub.get('name', 'Subteam')}",
        "url": team.get("url", "")
    } for team in top_teams for sub in team.get("subteams", [])]

    hackathon_items = [{"name": h["name"], "url": h.get("url", "")} for h in hackathons]
    course_items = [{"name": c["name"], "url": c.get("url", "")} for c in courses]

    try:
        reasons = get_reasons_from_llm(jd, team_items, hackathon_items, course_items)
    except Exception:
        return jsonify({"error": "Failed to fetch LLM reasons"}), 500

    def with_reasons(items, reason_map):
        return [{
            "name": x["name"],
            "reason": reason_map.get(x["name"], "Relevant opportunity."),
            "url": x.get("url") or opportunities_lookup.get(x["name"], {}).get("url", "")
        } for x in items]

    return jsonify({
        "student_teams": with_reasons(team_items, reasons.get("student_teams", {})),
        "hackathons": with_reasons(hackathon_items, reasons.get("hackathons", {})),
        "courses": with_reasons(course_items, reasons.get("courses", {}))
    })


if __name__ == "__main__":
    app.run(debug=True, port=3000)
