from sentence_transformers import SentenceTransformer
import json
import os

# Load data
here = os.path.dirname(__file__)
with open(os.path.join(here, "opportunities.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode student teams and subteams
teams = data.get("student_teams", [])
for team in teams:
    full_text = (
        team.get("name", "") + " "
        + team.get("description", "") + " "
        + " ".join(team.get("tags", []))
    )
    team["embedding"] = model.encode(full_text).tolist()

    subteams = team.get("subteams", [])
    if isinstance(subteams, list):
        for sub in subteams:
            if isinstance(sub, dict):
                sub_text = (
                    sub.get("name", "") + " "
                    + sub.get("focus", "") + " "
                    + " ".join(sub.get("tags", []))
                )
                sub["embedding"] = model.encode(sub_text).tolist()

# Encode hackathons
hackathons = data.get("hackathons", [])
for hack in hackathons:
    full_text = (
        hack.get("name", "") + " "
        + hack.get("description", "") + " "
        + " ".join(hack.get("tags", []))
    )
    hack["embedding"] = model.encode(full_text).tolist()

# Encode courses
courses = data.get("courses", [])
for course in courses:
    full_text = (
        course.get("name", "") + " "
        + course.get("description", "") + " "
        + " ".join(course.get("tags", []))
    )
    course["embedding"] = model.encode(full_text).tolist()

# Save output
with open(os.path.join(here, "teams_with_embeddings.json"), "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
