import streamlit as st
import pdfplumber
import pickle
import re
import numpy as np
import docx
import pandas as pd
from PIL import Image
import pytesseract

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Resume Analyzer", layout="centered")

# ---------------- SOFT MODERN UI ----------------
st.markdown("""
<style>

/* SOFT GRADIENT BACKGROUND (BLUE + TEAL + GREEN) */
.stApp {
    background: linear-gradient(135deg, #d0f4ff, #c7f9cc, #bde0fe);
}

/* TITLE */
.title {
    font-size: 36px;
    font-weight: 800;
    text-align: center;
    color: #1f2937;
    margin-bottom: 20px;
}

/* FILE UPLOADER (GLASS STYLE) */
section[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.6);
    padding: 15px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.8);
}

/* FILE UPLOADER TEXT FIX */
section[data-testid="stFileUploader"] * {
    color: #1f2937 !important;
    font-weight: 600;
}

/* BUTTON (SOFT BLUE-GREEN GRADIENT) */
.stButton>button {
    background: linear-gradient(135deg, #4facfe, #00f2fe);
    color: white;
    border-radius: 30px;
    padding: 10px 25px;
    font-weight: 700;
    border: none;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}

/* BUTTON HOVER */
.stButton>button:hover {
    transform: scale(1.05);
    transition: 0.3s;
}

/* RESULT CARD (GLASS MORPHISM LIGHT) */
.result-card {
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(18px);
    border-radius: 22px;
    padding: 25px;
    margin-top: 20px;
    color: #1f2937;
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    line-height: 1.9;
}

/* HIGHLIGHT TEXT */
.highlight {
    color: #0ea5e9;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)

# ---------------- FUNCTIONS ----------------

def extract_text(file):
    text = ""

    try:
        if file.type == "application/pdf":
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""

        elif file.type == "text/plain":
            text = file.read().decode("utf-8", errors="ignore")

        elif file.type == "text/csv":
            df = pd.read_csv(file)
            text = df.to_string()

        elif "word" in file.type:
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"

        elif "image" in file.type:
            img = Image.open(file)
            text = pytesseract.image_to_string(img)

    except:
        st.error("File read error")

    return text


def clean_text(text):
    return re.sub(r'[^a-zA-Z ]', ' ', text).lower()


def extract_name(text):
    for line in text.split("\n")[:5]:
        if re.match(r'^[A-Z][a-z]+(?: [A-Z][a-z]+)+$', line.strip()):
            return line.strip()
    return "Not Found"
 
def extract_education(text):
    text = text.lower()

    patterns = {
        "B.Tech": r"b\.?tech|bachelor.*technology",
        "M.Tech": r"m\.?tech|master.*technology",
        "B.Sc": r"b\.?sc|bachelor.*science",
        "M.Sc": r"m\.?sc|master.*science",
        "MBA": r"\bmba\b",
        "BCA": r"\bbca\b",
        "MCA": r"\bmca\b",
        "PhD": r"\bphd\b",
        "10th": r"\b10th\b|secondary",
        "12th": r"\b12th\b|senior secondary"
    }

    found = []

    for edu, pattern in patterns.items():
        if re.search(pattern, text):
            found.append(edu)

    return ", ".join(found) if found else "Not Mentioned"


def extract_skills(text):
    skills_db = [
        # -------- PROGRAMMING --------
        "python", "java", "c++", "c#", "javascript", "typescript",
        "php", "r programming", "go", "ruby",

        # -------- DATA / AI --------
        "machine learning", "deep learning", "nlp", "data science",
        "data analysis", "statistics", "pandas", "numpy",
        "tensorflow", "pytorch", "keras", "computer vision",

        # -------- DATABASE --------
        "sql", "mysql", "postgresql", "mongodb", "sqlite",

        # -------- VISUALIZATION --------
        "power bi", "tableau", "excel", "matplotlib", "seaborn",

        # -------- WEB DEVELOPMENT --------
        "html", "css", "react", "angular", "node.js", "django", "flask",

        # -------- CLOUD / DEVOPS --------
        "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",

        # -------- MOBILE --------
        "android development", "flutter", "react native",

        # -------- NON-TECH SKILLS --------
        "communication", "teamwork", "leadership", "problem solving",
        "time management", "critical thinking", "adaptability",
        "project management", "decision making"
    ]

    return [s for s in skills_db if s in text.lower()]
# ---------------- UI ----------------

st.markdown('<div class="title">Resume Analyzer</div>', unsafe_allow_html=True)

resume_file = st.file_uploader(
    "Upload Resume",
    type=["pdf","txt","csv","docx","png","jpg","jpeg"]
)

jd_file = st.file_uploader(
    "Upload Job Description",
    type=["pdf","txt","csv","docx"]
)

if st.button("Analyze Resume"):

    if resume_file:

        text = extract_text(resume_file)
        clean = clean_text(text)

        vec = vectorizer.transform([clean])
        prediction = model.predict(vec)[0]
        confidence = np.max(model.predict_proba(vec)) * 100

        name = extract_name(text)
        education = extract_education(text)
        skills = extract_skills(clean)

        score = 0
        missing = []

        if jd_file:
            jd_text = extract_text(jd_file)
            jd_clean = clean_text(jd_text)

            jd_skills = extract_skills(jd_clean)

            if jd_skills:
                match = len(set(skills) & set(jd_skills))
                score = int((match / len(jd_skills)) * 100)
                missing = list(set(jd_skills) - set(skills))

        # ---------------- OUTPUT ----------------
        st.markdown(f"""
        <div class="result-card">
            <p><b>Name:</b> <span class="highlight">{name}</span></p>
            <p><b>Education:</b> {education}</p>
            <p><b>Skills:</b> {", ".join(skills)}</p>
            <p><b>Predicted Role:</b> {prediction}</p>
            <p><b>Confidence:</b> {confidence:.2f}%</p>
            <p><b>Experience:</b> Fresher</p>
            <p><b>Match Score:</b> <span class="highlight">{score}%</span></p>
            <p><b>Eligibility:</b> {"Eligible" if score > 70 else "Not Eligible"}</p>
            <p><b>Missing Skills:</b> {", ".join(missing) if missing else "None"}</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Please upload a resume first")