# AI Job Match & Application Assistant — Tavily + Groq + Streamlit

## Architecture

Google Colab / Python
-> Resume PDF + User Input
-> Resume Analyzer + Role/Location
-> Tavily Job Search
-> Job URLs
-> Tavily Extract
-> Job Description
-> Groq JD Analyzer
-> Required Skills
-> Groq Matcher
-> Match Score
-> Groq Application Writer
-> LinkedIn + Email + Cover Letter

## Features

- Upload PDF, DOCX or TXT resume.
- Extract resume text.
- Analyze resume with Groq.
- Search public job listings on:
  - LinkedIn
  - Indeed
  - Naukri
- Extract job-page content with Tavily Extract.
- Analyze every JD with Groq.
- Compare candidate profile with required skills.
- Generate a 0–100 match score.
- Show matched skills and skill gaps.
- Select top 5 or 6 jobs.
- Generate a customized cover letter for each job.
- Extract recruitment-style public email addresses when present in the extracted page.
- Open a prefilled Gmail compose window for the generated email.
- No automatic application submission.

## Setup

### 1. Create environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create API keys

You need:

- Tavily API key
- Groq API key

You can enter both directly in the Streamlit sidebar, or create `.env` from `.env.example`.

Example:

```text
TAVILY_API_KEY=...
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
```

### 4. Run

```bash
streamlit run app.py
```

## Streamlit Cloud deployment

Upload this project to GitHub.

Then deploy the repository with Streamlit Community Cloud.

Add these values under the app's Secrets:

```toml
TAVILY_API_KEY = "your_key"
GROQ_API_KEY = "your_key"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

Do not commit `.env` or API keys to GitHub.

## Important job-search note

This MVP uses public web-search results through Tavily. LinkedIn, Indeed and Naukri can restrict automated access and scraping. For a production application, use authorized/licensed job-search data sources and follow each platform's terms and robots/access rules.

Search results can also contain expired listings or pages that are not directly accessible. The app therefore displays the original listing URL so the user can verify it.

## Gmail behavior

The app uses a prefilled Gmail Compose URL rather than silently sending email.

This means the user can review:
- recipient
- subject
- email body
- job URL

before clicking Send.

For a true Gmail API "saved Draft" workflow, add Google OAuth and the Gmail API later. This is intentionally separated from the core matching pipeline so the Streamlit app can be deployed more easily.

## Recommended project title

**AI-Powered Job Recommendation and Application Assistant Using Resume–JD Matching**

Alternative:

**Intelligent Job Matching and Automated Application Assistant**

## Recommended technologies

- Python
- Streamlit
- Tavily Search API
- Tavily Extract API
- Groq LLM
- Llama model
- PyPDF
- python-docx
- NLP / Information Extraction
- Semantic/LLM-based Resume–JD Matching

## Future improvements

1. Embedding-based semantic similarity.
2. Skill ontology and synonym mapping.
3. Salary extraction.
4. Experience-level filtering.
5. Remote/hybrid/on-site filtering.
6. Job freshness filtering.
7. Company information extraction.
8. Resume tailoring per job.
9. PDF cover-letter generation.
10. Application tracking dashboard.
11. Database storage.
12. User authentication.
13. Authorized Gmail API draft creation.
14. Better company/recruiter contact discovery using permitted sources.
