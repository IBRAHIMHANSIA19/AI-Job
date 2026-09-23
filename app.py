import os
import re
import json
import base64
from email.message import EmailMessage
from urllib.parse import quote_plus

import streamlit as st
from pypdf import PdfReader
from docx import Document
from tavily import TavilyClient
from groq import Groq

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

st.set_page_config(
    page_title="AI Job Matcher",
    page_icon="💼",
    layout="wide",
)

# ============================================================
# CONFIG / CLIENTS
# ============================================================

def get_secret(name):
    value = os.getenv(name, "")
    try:
        if not value:
            value = st.secrets.get(name, "")
    except Exception:
        pass
    return value


def get_tavily():
    key = st.session_state.get("TAVILY_API_KEY") or get_secret("TAVILY_API_KEY")
    if not key or key.strip() in ["your_tavily_api_key", "PASTE_YOUR_TAVILY_KEY", "your_tavily_api_key_here"]:
        raise RuntimeError("TAVILY_API_KEY is missing or invalid. Please add your key to the .env file.")
    return TavilyClient(api_key=key)


def get_groq():
    key = st.session_state.get("GROQ_API_KEY") or get_secret("GROQ_API_KEY")
    if not key or key.strip() in ["your_groq_api_key", "PASTE_YOUR_GROQ_KEY", "your_groq_api_key_here"]:
        raise RuntimeError("GROQ_API_KEY is missing or invalid. Please add your key to the .env file.")
    return Groq(api_key=key, timeout=30.0, max_retries=3)


AVAILABLE_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


def groq_model():
    return (
        st.session_state.get("GROQ_MODEL")
        or get_secret("GROQ_MODEL")
        or "openai/gpt-oss-120b"
    )


def run_groq_chat(client, messages, temperature=0, json_mode=True):
    try:
        live_models = [
            m.id for m in client.models.list().data 
            if not m.id.startswith("whisper") and "guard" not in m.id.lower()
        ]
    except Exception:
        live_models = []

    selected_model = groq_model()
    all_candidates = [selected_model] + live_models + AVAILABLE_GROQ_MODELS
    seen = set()
    models_to_try = []
    for m in all_candidates:
        if m and m not in seen:
            seen.add(m)
            models_to_try.append(m)

    last_exception = None
    for model_name in models_to_try:
        try:
            kwargs = {
                "model": model_name,
                "temperature": temperature,
                "messages": messages,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            st.session_state["GROQ_MODEL"] = model_name
            return response
        except Exception as exc:
            err_str = str(exc).lower()
            if any(k in err_str for k in ["decommissioned", "model_not_found", "does not exist", "404", "400", "invalid_request_error", "not supported"]):
                last_exception = exc
                continue
            raise exc

    if last_exception:
        raise last_exception


# ============================================================
# RESUME ANALYZER
# ============================================================

def extract_resume_text(uploaded_file):
    if uploaded_file is None:
        return ""

    data = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        import io
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        import io
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    return ""


def resume_analyzer(resume_text):
    client = get_groq()

    prompt = f"""
Analyze this resume for a job-matching system.

Return ONLY valid JSON with this structure:
{{
  "candidate_name": "",
  "education": [],
  "experience": [],
  "skills": [],
  "tools_and_technologies": [],
  "certifications": [],
  "projects": [],
  "target_roles": [],
  "years_experience": 0
}}

Rules:
- Use only information present in the resume.
- Do not invent facts.
- Normalize obvious skill variations where safe, e.g. "MS Excel" -> "Excel".
- Keep the result concise.

RESUME:
{resume_text[:18000]}
"""

    try:
        response = run_groq_chat(
            client=client,
            messages=[
                {"role": "system", "content": "You are a precise resume information extraction engine."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            json_mode=True,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as exc:
        err_msg = str(exc)
        if "Connection error" in err_msg or "connect" in err_msg.lower() or "ConnectionRefused" in err_msg:
            raise RuntimeError(
                "Connection Error to Groq API. Please check your internet connection or verify your GROQ_API_KEY in .env file."
            ) from exc
        raise RuntimeError(f"Resume analysis failed: {err_msg}") from exc


# ============================================================
# TAVILY JOB SEARCH
# ============================================================

PLATFORMS = {
    "LinkedIn": "linkedin.com/jobs",
    "Indeed": "indeed.com",
    "Naukri": "naukri.com",
}


def search_jobs(role, location, max_per_platform=5):
    tavily = get_tavily()
    results = []

    for platform, domain in PLATFORMS.items():
        query = (
            f'site:{domain} "{role}" "{location}" '
            f'(job OR jobs OR hiring OR vacancy)'
        )

        response = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=max_per_platform,
            include_answer=False,
            include_raw_content=False,
        )

        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "platform": platform,
            })

    # Deduplicate URLs.
    seen = set()
    unique = []
    for job in results:
        url = job["url"].split("#")[0].strip()
        if not url or url in seen:
            continue
        seen.add(url)
        job["url"] = url
        unique.append(job)

    return unique


def extract_job_page(url):
    tavily = get_tavily()

    try:
        response = tavily.extract(
            urls=[url],
            include_images=False,
        )

        results = response.get("results", [])
        if results:
            item = results[0]
            return item.get("raw_content") or item.get("content") or ""

    except Exception as exc:
        return f"[Extraction failed: {exc}]"

    return ""


# ============================================================
# JD ANALYZER
# ============================================================

def jd_analyzer(job):
    client = get_groq()

    jd_text = job.get("description", "")[:16000]

    prompt = f"""
Analyze this job description.

Return ONLY valid JSON:
{{
  "job_title": "",
  "company": "",
  "location": "",
  "experience_required": "",
  "employment_type": "",
  "required_skills": [],
  "preferred_skills": [],
  "responsibilities": [],
  "keywords": []
}}

Rules:
- Extract facts only from the supplied job description.
- If a field is unavailable, use an empty string or [].
- Do not invent company information.

JOB DESCRIPTION:
{jd_text}
"""

    response = run_groq_chat(
        client=client,
        messages=[
            {"role": "system", "content": "You are a job-description analysis engine."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        json_mode=True,
    )
    return json.loads(response.choices[0].message.content)


# ============================================================
# GROQ MATCHER
# ============================================================

def matcher(resume_profile, jd_profile):
    client = get_groq()

    prompt = f"""
Compare this candidate profile with this job profile.

Return ONLY valid JSON:
{{
  "match_score": 0,
  "matched_skills": [],
  "missing_skills": [],
  "matching_reasons": [],
  "skill_gap_summary": "",
  "recommendation": "Strong Match | Moderate Match | Low Match"
}}

Rules:
- Score from 0 to 100.
- Base the score on skills, education, experience, tools, and role alignment.
- Do not infer skills that are not supported by the candidate profile.
- Be consistent and explain the score through the returned fields.

CANDIDATE:
{json.dumps(resume_profile, ensure_ascii=False)}

JOB:
{json.dumps(jd_profile, ensure_ascii=False)}
"""

    response = run_groq_chat(
        client=client,
        messages=[
            {"role": "system", "content": "You are a deterministic recruitment matching engine."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        json_mode=True,
    )
    return json.loads(response.choices[0].message.content)


# ============================================================
# APPLICATION WRITER
# ============================================================

def application_writer(resume_profile, job, jd_profile, match_profile):
    client = get_groq()

    prompt = f"""
Write a tailored job application package.

Return ONLY valid JSON:
{{
  "cover_letter": "",
  "email_subject": "",
  "email_body": ""
}}

Candidate:
{json.dumps(resume_profile, ensure_ascii=False)}

Job:
{json.dumps({
    "title": job.get("title"),
    "platform": job.get("platform"),
    "url": job.get("url")
}, ensure_ascii=False)}

Job profile:
{json.dumps(jd_profile, ensure_ascii=False)}

Match:
{json.dumps(match_profile, ensure_ascii=False)}

Rules:
- Never invent qualifications, projects, employers, achievements, or experience.
- Use only candidate information supplied above.
- Cover letter: 250-350 words.
- Email body: concise, professional, 120-180 words.
- Use "Dear Hiring Manager" unless a real hiring person's name is explicitly available.
- Do not claim that the candidate already applied.
"""

    response = run_groq_chat(
        client=client,
        messages=[
            {"role": "system", "content": "You write accurate professional job applications."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        json_mode=True,
    )
    return json.loads(response.choices[0].message.content)


# ============================================================
# EMAIL EXTRACTION
# ============================================================

def extract_public_emails(text):
    emails = re.findall(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        text or "",
    )

    generic_words = [
        "hr", "career", "careers", "recruit", "recruitment",
        "jobs", "hiring", "talent", "people"
    ]

    output = []
    for email in emails:
        local = email.split("@")[0].lower()
        if any(word in local for word in generic_words):
            output.append(email)

    return sorted(set(output))


def gmail_compose_url(to_email, subject, body):
    return (
        "https://mail.google.com/mail/?view=cm&fs=1"
        f"&to={quote_plus(to_email)}"
        f"&su={quote_plus(subject)}"
        f"&body={quote_plus(body)}"
    )


# ============================================================
# UI
# ============================================================

st.title("💼 AI Job Match & Application Assistant")
st.write(
    "Resume → Job Search → JD Extraction → JD Analysis → AI Matching → "
    "Cover Letter → Email Draft"
)

with st.sidebar:
    st.header("⚙️ Configuration")

    location = st.text_input("Preferred Location", "India")
    final_jobs = st.slider("Number of jobs", 1, 10, 5)

    selected_m = st.selectbox(
        "Groq AI Model",
        AVAILABLE_GROQ_MODELS,
        index=0,
    )
    st.session_state["GROQ_MODEL"] = selected_m

    with st.expander("🔑 API Key Setup (Optional)", expanded=False):
        st.caption("Enter keys here if not configured in `.env` file:")
        t_key = st.text_input("Tavily API Key", type="password", key="tav_input")
        g_key = st.text_input("Groq API Key", type="password", key="groq_input")
        if t_key:
            st.session_state["TAVILY_API_KEY"] = t_key
        if g_key:
            st.session_state["GROQ_API_KEY"] = g_key

# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

st.header("1️⃣ Resume + Job Input")

resume_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx", "txt"],
)

col1, col2 = st.columns(2)

with col1:
    target_role = st.text_input(
        "Target Role",
        placeholder="e.g. Data Analyst Intern",
    )

with col2:
    st.write("")
    st.write("")
    search_info = st.info(
        "Job search uses LinkedIn, Indeed and Naukri public web results."
    )

job_description = st.text_area(
    "Optional Job Description",
    height=180,
    placeholder="Paste a JD if you already have one. The app can also extract JDs from discovered job URLs.",
)

resume_text = ""
resume_profile = None

if resume_file:
    resume_text = extract_resume_text(resume_file)

    if resume_text:
        st.success(f"Resume loaded: {resume_file.name}")

        with st.expander("Preview Resume Text"):
            st.text(resume_text[:6000])

        if st.button("🧠 Analyze Resume", use_container_width=True):
            groq_key_val = st.session_state.get("GROQ_API_KEY") or get_secret("GROQ_API_KEY")
            if not groq_key_val or groq_key_val.strip() in ["your_groq_api_key", "PASTE_YOUR_GROQ_KEY", "your_groq_api_key_here"]:
                st.error("⚠️ GROQ_API_KEY is missing or invalid. Please add your valid GROQ_API_KEY in the `.env` file.")
            else:
                with st.spinner("Analyzing resume with Groq..."):
                    try:
                        st.session_state["resume_profile"] = resume_analyzer(resume_text)
                        st.success("Resume analysis completed.")
                    except Exception as e:
                        st.error(f"❌ Resume analysis failed: {e}")

if "resume_profile" in st.session_state:
    with st.expander("Resume Analyzer Output", expanded=False):
        st.json(st.session_state["resume_profile"])

# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

if st.button("🚀 Find & Analyze Top Jobs", type="primary", use_container_width=True):
    if not resume_text:
        st.error("Upload a resume first.")
        st.stop()

    if not target_role:
        st.error("Enter a target role.")
        st.stop()

    if not (st.session_state.get("TAVILY_API_KEY") or get_secret("TAVILY_API_KEY")):
        st.error("Add your Tavily API key.")
        st.stop()

    if not (st.session_state.get("GROQ_API_KEY") or get_secret("GROQ_API_KEY")):
        st.error("Add your Groq API key.")
        st.stop()

    try:
        with st.status("Running AI job pipeline...", expanded=True) as status:

            # 1. Resume Analyzer
            st.write("📄 Resume Analyzer")
            if "resume_profile" not in st.session_state:
                resume_profile = resume_analyzer(resume_text)
                st.session_state["resume_profile"] = resume_profile
            else:
                resume_profile = st.session_state["resume_profile"]

            # 2. Tavily Job Search
            st.write("🔎 Tavily Job Search")
            jobs = search_jobs(
                target_role,
                location,
                max_per_platform=5,
            )

            if not jobs:
                status.update(label="No jobs found.", state="error")
                st.error("No public listings were found. Try another role/location.")
                st.stop()

            # 3. Tavily Extract
            st.write("🌐 Tavily Job Description Extraction")
            extracted_jobs = []

            for job in jobs:
                description = job.get("snippet", "")
                extracted = extract_job_page(job["url"])

                if extracted and not extracted.startswith("[Extraction failed:"):
                    description = extracted

                job["description"] = description
                extracted_jobs.append(job)

            # 4. Groq JD Analyzer
            st.write("🧠 Groq JD Analyzer")
            analyzed = []

            for job in extracted_jobs:
                try:
                    job["jd_profile"] = jd_analyzer(job)
                    analyzed.append(job)
                except Exception as exc:
                    job["jd_profile"] = {
                        "job_title": job.get("title", ""),
                        "company": "",
                        "location": "",
                        "experience_required": "",
                        "employment_type": "",
                        "required_skills": [],
                        "preferred_skills": [],
                        "responsibilities": [],
                        "keywords": [],
                        "_error": str(exc),
                    }
                    analyzed.append(job)

            # 5. Groq Matcher
            st.write("📊 Groq Matcher")
            for job in analyzed:
                job["match_profile"] = matcher(
                    resume_profile,
                    job["jd_profile"],
                )

            analyzed.sort(
                key=lambda x: float(x["match_profile"].get("match_score", 0)),
                reverse=True,
            )

            top_jobs = analyzed[:final_jobs]

            # 6. Application Writer
            st.write("✍️ Groq Application Writer")
            for job in top_jobs:
                job["application"] = application_writer(
                    resume_profile,
                    job,
                    job["jd_profile"],
                    job["match_profile"],
                )

            st.session_state["jobs"] = top_jobs
            status.update(
                label=f"Completed — {len(top_jobs)} jobs selected.",
                state="complete",
            )

    except Exception as e:
        st.error(f"Pipeline failed: {e}")

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

if "jobs" in st.session_state:
    jobs = st.session_state["jobs"]

    st.divider()
    st.header(f"🎯 Top {len(jobs)} Job Matches")

    # Quick View Section for Links/URLs
    with st.expander(f"🔗 Quick View: Direct Application Links (Top {len(jobs)} Jobs)", expanded=True):
        for idx, j in enumerate(jobs, start=1):
            title = j.get('jd_profile', {}).get('job_title') or j.get('title', 'Job Listing')
            platform = j.get('platform', '')
            score = j.get('match_profile', {}).get('match_score', 0)
            url = j.get('url', '')
            st.markdown(f"**{idx}. [{title}]({url})** | Score: `{score}%` | *{platform}*  \n👉 **Apply URL:** [{url}]({url})")

    st.markdown("---")

    for index, job in enumerate(jobs, start=1):
        match = job["match_profile"]
        jd_info = job["jd_profile"]
        app = job["application"]

        with st.container(border=True):
            st.subheader(
                f"{index}. {jd_info.get('job_title') or job.get('title')}"
            )

            c1, c2, c3 = st.columns([2, 2, 1])

            with c1:
                st.write(f"**Platform:** {job['platform']}")
                st.write(
                    f"**Company:** {jd_info.get('company') or 'Not available'}"
                )

            with c2:
                st.write(
                    f"**Location:** {jd_info.get('location') or location}"
                )
                st.write(
                    f"**Experience:** "
                    f"{jd_info.get('experience_required') or 'Not specified'}"
                )

            with c3:
                st.metric(
                    "Match Score",
                    f"{match.get('match_score', 0)}%",
                )

            if job.get("url"):
                st.markdown(f"🔗 **Direct Application URL:** [{job['url']}]({job['url']})")
                st.link_button("🚀 Open & Apply Now", job["url"])

            st.markdown("### 🧩 Match Analysis")

            m1, m2 = st.columns(2)

            with m1:
                st.write("**Matched Skills**")
                st.write(", ".join(match.get("matched_skills", [])) or "None identified")

            with m2:
                st.write("**Missing / Skill Gaps**")
                st.write(", ".join(match.get("missing_skills", [])) or "None identified")

            st.write(
                f"**Summary:** {match.get('skill_gap_summary', '')}"
            )

            with st.expander("Required Skills"):
                st.write(
                    ", ".join(jd_info.get("required_skills", []))
                    or "Not extracted"
                )

            st.markdown("### 📝 Tailored Cover Letter")
            cover = app.get("cover_letter", "")
            st.text_area(
                "Cover Letter",
                cover,
                height=320,
                key=f"cover_{index}",
            )

            st.download_button(
                "⬇️ Download Cover Letter",
                data=cover,
                file_name=f"cover_letter_{index}.txt",
                mime="text/plain",
                key=f"download_{index}",
            )

            st.markdown("### 📧 HR Email")

            emails = extract_public_emails(job.get("description", ""))

            if emails:
                st.success(
                    "Public recruitment-style email(s) found: "
                    + ", ".join(emails)
                )
                default_email = emails[0]
            else:
                st.info(
                    "No public HR/recruitment email was confidently extracted "
                    "from this job page."
                )
                default_email = ""

            hr_email = st.text_input(
                "HR / Recruiter Email",
                value=default_email,
                key=f"email_{index}",
            )

            subject = app.get(
                "email_subject",
                f"Application for {jd_info.get('job_title') or job.get('title')}",
            )
            body = app.get("email_body", "")

            st.text_input(
                "Email Subject",
                value=subject,
                key=f"subject_{index}",
            )

            st.text_area(
                "Email Body",
                body,
                height=230,
                key=f"body_{index}",
            )

            if hr_email:
                st.link_button(
                    "📨 Open Gmail Compose",
                    gmail_compose_url(
                        hr_email,
                        subject,
                        body,
                    ),
                )

            st.caption(
                "Review the job listing, recipient, match score, and generated "
                "application before sending. The app does not auto-submit applications."
            )
