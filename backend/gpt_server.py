from flask import Flask, request, jsonify
from flask_cors import CORS
import json, os, re
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

# Build a lookup table from name to team/subteam info
opportunities_lookup = {}

for team in opportunities["student_teams"]:
    for st in team.get("subteams", []):
        if not isinstance(st, dict):
            print(f"⚠️ Skipping malformed subteam in {team['name']}: {st}")
            continue
        full_name = f"{team['name']} – {st.get('name', 'Subteam')}"
        opportunities_lookup[full_name] = {
            "url": team.get("url", "")
        }

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

def summarize_recommendations(teams, hackathons, courses):
    def short(team, st):
        return {
            "name": f"{team['name']} – {st.get('name', 'Subteam')}",
            "reason": st.get("reason", ""),
            "url": team.get("url", "")
        }

    summarized_teams = []
    for team in teams:
        for st in team.get("subteams", []):
            summarized_teams.append(short(team, st))

    return {
        "student_teams": summarized_teams[:5],
        "hackathons": hackathons[:3],
        "courses": courses[:3]
    }

def truncate_list(items, max_items=5, max_chars=1000):
    text_lines = []
    for item in items[:max_items]:
        if isinstance(item, dict):
            text_lines.append(f"- {item.get('name', '')}: {item.get('reason', '')}")
        else:
            text_lines.append(str(item))

    text = '\n'.join(text_lines)
    return text[:max_chars]



def build_prompt(job_desc, summary):
    student_teams = truncate_list(summary['student_teams'])
    hackathons = truncate_list(summary['hackathons'])
    courses = truncate_list(summary['courses'])

    return f"""You are an assistant recommending opportunities based on the following job description:

Job Description:
{job_desc}

Top Matching Student Clubs:
{student_teams}

Top Hackathons:
{hackathons}

Top Courses:
{courses}

Return ONLY a valid JSON object with exactly these keys: "student_teams", "hackathons", and "courses".

Each key must map to a list of objects with:
- "name": string
- "reason": 1-line reason it's relevant (string)
- "url": a direct clickable link (string)

End your response immediately after the final }}. Do not continue or add extra text.

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
If you cannot find any relevant matches, return empty lists. Your entire response MUST be a valid JSON object and nothing else.
"""



def extract_first_complete_json(text):
    start = text.find('{')
    if start == -1:
        return None

    stack = []
    for i in range(start, len(text)):
        if text[i] == '{':
            stack.append('{')
        elif text[i] == '}':
            stack.pop()
            if not stack:
                return text[start:i+1]
    return None

def summarize_recommendations(teams, hackathons, courses):
    return {
        "student_teams": teams,
        "hackathons": hackathons,
        "courses": courses
    }



def get_reasons_from_llm(job_desc, teams, hackathons, courses):
    def format_list(items):
        return "\n".join(f"- {item['name']}" for item in items)

    prompt = f"""You are an assistant. Based on this job description:

{job_desc}

Explain briefly why each of these are relevant opportunities. Return a JSON object mapping each name to a short reason.

Student Teams:
{format_list(teams)}

Hackathons:
{format_list(hackathons)}

Courses:
{format_list(courses)}

Example:

{{
  "student_teams": {{
    "Team A": "Reason for Team A",
    "Team B": "Reason for Team B"
  }},
  "hackathons": {{
    "Hackathon 1": "Reason for Hackathon 1"
  }},
  "courses": {{
    "Course X": "Reason for Course X"
  }}
}}

Respond with a JSON object only, in this format: ```json { ... } ```
"""

    payload = {
        "model": "wizardlm-2-7b",
        "messages": [
            {"role": "system", "content": "You are an assistant that provides short reasons why opportunities are relevant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 2048
    }

    response = requests.post(
        "http://127.0.0.1:1234/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=30
    )
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    content = re.sub(r"^```json|```$", "", content.strip())

    # Print for debugging
    print("🔎 Raw LLM content:", content)

    try:
        return extract_json(content)
    except Exception as e:
        print("❌ Failed to extract JSON:", e)
        print("🔁 Raw content:", content)
        raise

def extract_json(content: str):
    # Try to extract JSON block from triple backticks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Fallback: try to find first valid JSON object in text
    match = re.search(r"(\{.*\})", content, re.DOTALL)
    try:
        return json.loads(match.group(1))
    except Exception:
        raise ValueError("Could not find valid JSON block in LLM output. Full output:\n" + content)


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
    # Prepare lists without reasons for LM input
    teams_no_reason = []
    for team in top_teams:
        # each team has subteams, flatten subteams with team name combined for clarity
        for st in team.get("subteams", []):
            teams_no_reason.append({
            "name": f"{team['name']} – {st.get('name', 'Subteam')}",
            "url": team.get("url", "")
        })


    hackathons_no_reason = [{"name": h["name"], "url": h.get("url", "")} for h in top_hackathons]
    courses_no_reason = [{"name": c["name"], "url": c.get("url", "")} for c in filtered_courses]


    # Call LM to get reasons only
    reasons = get_reasons_from_llm(jd, teams_no_reason, hackathons_no_reason, courses_no_reason)

    # Merge reasons with original data (and real URLs)
    def merge_reasons(items, reasons_dict):
        merged = []
        for item in items:
            reason = reasons_dict.get(item["name"], "")
            url = item.get("url") or opportunities_lookup.get(item["name"], {}).get("url", "")
            merged.append({
                "name": item["name"],
                "reason": reason,
                "url": url
            })
        return merged


    student_teams_out = merge_reasons(teams_no_reason, reasons.get("student_teams", {}))
    hackathons_out = merge_reasons(hackathons_no_reason, reasons.get("hackathons", {}))
    courses_out = merge_reasons(courses_no_reason, reasons.get("courses", {}))

    # Final JSON to return
    result = {
        "student_teams": student_teams_out,
        "hackathons": hackathons_out,
        "courses": courses_out
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=3000)
