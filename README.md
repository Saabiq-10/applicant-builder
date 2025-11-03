<img width="48" height="48" alt="ab_logo48" src="https://github.com/user-attachments/assets/a6dde2f9-3a0c-4890-9631-fa2b7c663b20" />

# Applicant Builder for LinkedIn

An AI-powered Chrome extension that reads a job description and recommends student teams, upcoming hackathons, and online courses that build the skills in that job posting.
The goal is to help engineering students take action fast instead of scrolling for hours to stay up to date with opportunities and guessing what to join.

---

## Purpose

Most students want to join meaningful projects but spend too much time searching for the right ones.
Applicant Builder reverses that process and brings opportunities directly to you.

It reads the description of your dream job and lists student teams, courses, and hackathons that match the skills employers look for.
Each suggestion includes a link and a short reason why it fits, so you can act immediately.

The main goal is **club discovery**, helping students find opportunities they might have missed.
The secondary goal is **planning**, giving a clear path to build experience for specific roles.
It makes getting involved simple, fast, and personal.

---

## Features

* Smart recommendations: Picks the most relevant TMU student teams, subteams, hackathons, and courses.
* Context-aware matching: Uses SentenceTransformers embeddings to compare the job description with each opportunity.
* Structured data sources: Pulls URLs and skill tags from `opportunities.json` and `teams_with_embeddings.json`, not from the language model.
* Reliable backend: Flask server parses model output and fills gaps so the extension does not crash.
* Clean UI: Each suggestion includes a short explanation under it so you understand why it matches that job.
* Upcoming hackathons: Sorted by soonest date so you know what to sign up for next.

---

## Example Output

Below is a live example of the Chrome extension running on a real LinkedIn job post.  
The panel on the right shows AI-generated recommendations for student teams, hackathons, and courses based on the job description.

<img width="1961" height="1376" alt="Screenshot 2025-11-02 232635" src="https://github.com/user-attachments/assets/2cbca79e-ed69-4394-84f3-17a38973b27a" />
<img width="1956" height="1380" alt="Screenshot 2025-11-02 232713" src="https://github.com/user-attachments/assets/d440d6fb-2217-4df2-93c1-4fa33ccc6d11" />

---

## Tech Stack

* **Frontend:** JavaScript, HTML, CSS (Chrome Extension)
* **Backend:** Python (Flask, SentenceTransformers, LM Studio API)
* **Data:** JSON-based opportunity database with embedded vectors
* **Environment:** Local inference (runs on CPU, can use GPU if available)

---

## Architecture

```text
📁 applicant-builder/
 ├── Body/                          # Chrome extension frontend
 │   ├── popup.html, popup.js       # Popup UI and API calls
 │   ├── content.js                 # Job scraping logic from LinkedIn
 │   └── manifest.json              # Extension configuration
 ├── backend/                       # Flask server (API endpoints + AI logic)
 │   ├── gpt_server.py              # Main backend
 │   ├── embed_teams.py             # Generates embeddings from raw data
 │   ├── opportunities.json         # Human-readable source data (clubs, subs, hackathons, courses)
 │   ├── teams_with_embeddings.json # Same data with embedding vectors added
 │   └── requirements.txt           # Dependencies
 └── LICENSE
```

---

## LM Studio Setup – Local AI

**LM Studio** is a free desktop app that runs large language models (LLMs) locally on your computer.
Applicant Builder uses it to generate short explanations for each recommendation without sending data to the cloud.

1. Download **LM Studio** from [lmstudio.ai](https://lmstudio.ai).
2. Load a local chat model such as `WizardLM-2-7B`.
3. Start the LM Studio local API server on port 1234.
4. The backend connects to this server to get reasoning output for each suggestion.

You can use any local model that supports an OpenAI-style `/v1/chat/completions` endpoint.

---

## Backend Setup

1. Clone this repository

```bash
git clone https://github.com/Saabiq-10/applicant-builder.git
cd applicant-builder
```

2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

3. Start the Flask server

```bash
python backend/gpt_server.py
```

The backend runs locally at:
`http://127.0.0.1:3000`

---

## Chrome Extension Setup

1. Open Chrome → Extensions → Manage Extensions.
2. Turn on Developer Mode.
3. Click "Load unpacked" and select the `Body/` folder.

Now open a LinkedIn job post, then open the extension popup.
You will see recommended teams, hackathons, and courses.

---

## Performance

* Runs fully local. No cloud calls.
* Uses CPU by default.
* If PyTorch can see CUDA on your machine, it will use GPU for faster embedding and inference.

You can check CUDA with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Adapting This for Your School

This repo is preloaded with Toronto Metropolitan University (TMU) engineering teams and events.
If you are not at TMU, you can still use this. You only have to edit the data and regenerate embeddings.

Step 1. Edit `backend/opportunities.json`

* Update `student_teams`, `subteams`, `hackathons`, and `courses`.
* Keep the same keys:

  * `name`, `description`, `tags`, `url`, etc.
  * Subteams should list focus areas, tools used, and what you do there.
* Replace TMU clubs with your own clubs.
* Replace Discord links with public links you are comfortable sharing.

Step 2. Generate fresh embeddings
Run:

```bash
python backend/embed_teams.py
```

What this does:

* Loads your updated `opportunities.json`.
* Uses the `all-MiniLM-L6-v2` SentenceTransformer model to embed each team, subteam, hackathon, and course as a vector.
* Saves everything to `teams_with_embeddings.json`. 

Step 3. Start the server again
`gpt_server.py` reads `teams_with_embeddings.json` when ranking and scoring opportunities against a job description.

After that, the Chrome extension should work the same way, but now it is tuned to your ecosystem.

---

## Example Flow

1. You open a Quality Engineering Intern at Rocket Lab on LinkedIn.
2. The extension scrapes the job description in the active tab.
3. The Flask backend embeds the job text, compares it to each subteam, hackathon, and course vector, and picks the best matches.
4. The backend also queries the local LLM (via LM Studio) for short human-readable reasons.
5. The popup shows:

   * which team or subteam to join,
   * which hackathon is coming up next,
   * which online course to take to close a skill gap.

---

## Limitations

Recommendations are limited to the data stored in `opportunities.json`.  
The system ranks and matches from the available options but does not fetch new opportunities from the web.  
To expand coverage, add more teams, hackathons, or courses to the JSON file and regenerate embeddings.

---

## License

Licensed under the GNU General Public License v3.0 (GPLv3).

You are allowed to use, study, modify, and share this code, including for commercial use.
If you distribute a modified version, you must also share the source under GPLv3 and keep the same freedoms for others.

---
