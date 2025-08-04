import os
import joblib
import pandas as pd
from scipy.sparse import hstack

# === Model paths ===
MODEL_PATH = "/Users/admin/Desktop/Projects/MindCV/models"

# === Load TF-IDF vectorizers ===
pos_tfidf = joblib.load(os.path.join(MODEL_PATH, "pos_tfidf.pkl"))
soft_tfidf = joblib.load(os.path.join(MODEL_PATH, "soft_tfidf.pkl"))
summ_tfidf = joblib.load(os.path.join(MODEL_PATH, "summ_tfidf.pkl"))
proj_tfidf = joblib.load(os.path.join(MODEL_PATH, "proj_tfidf.pkl"))

# === Load models and label encoders ===
logreg_mbti = joblib.load(os.path.join(MODEL_PATH, "logreg_mbti_group.pkl"))
logreg_belbin = joblib.load(os.path.join(MODEL_PATH, "logreg_belbinrole.pkl"))
le_mbti = joblib.load(os.path.join(MODEL_PATH, "labelencoder_mbti_group.pkl"))
le_belbin = joblib.load(os.path.join(MODEL_PATH, "labelencoder_belbinrole.pkl"))


def transform_sections(df: pd.DataFrame):
    """
    Transforms a DataFrame with columns: Position, SoftSkills, Summary, Projects
    into a combined sparse feature matrix.
    """
    pos_vec = pos_tfidf.transform(df['Position'])
    soft_vec = soft_tfidf.transform(df['SoftSkills'])
    summ_vec = summ_tfidf.transform(df['Summary'])
    proj_vec = proj_tfidf.transform(df['Projects'])

    return hstack([pos_vec, soft_vec, summ_vec, proj_vec])


def predict_from_sections(sections: dict) -> dict:
    """
    Runs prediction based on extracted resume sections.
    Returns MBTI and Belbin role predictions.
    """
    for key in ["Position", "SoftSkills", "Summary", "Projects"]:
        if not sections.get(key, "").strip():
            sections[key] = "empty"

    df = pd.DataFrame([sections])
    X_transformed = transform_sections(df)

    y_mbti = logreg_mbti.predict(X_transformed)
    y_belbin = logreg_belbin.predict(X_transformed)

    return {
        "mbti": le_mbti.inverse_transform(y_mbti)[0],
        "belbin": le_belbin.inverse_transform(y_belbin)[0]
    }


def run_inference_from_text(position, soft_skills, summary, projects):
    """
    Runs prediction using manually entered resume sections.
    """
    sections = {
        "Position": position,
        "SoftSkills": soft_skills,
        "Summary": summary,
        "Projects": projects
    }
    return predict_from_sections(sections)


# === Import PDF loader and NER extractor ===
from src.data_loader import extract_text
from src.ner_extractor import extract_sections_ner as extract_sections


def run_inference_from_file(file_path):
    """
    Extracts text from a PDF file, parses sections, and returns prediction.
    """
    raw_text = extract_text(file_path)
    print("📄 Raw text preview:\n", raw_text[:500], "...\n")

    sections = extract_sections(raw_text)
    print("✂️ Extracted sections:")
    for k, v in sections.items():
        print(f"— {k}: {v[:1000]}{'...' if len(v) > 1000 else ''}")

    return predict_from_sections(sections)


def run_inference_from_text_only(text: str):
    """
    Runs prediction from raw plain text (without a file).
    """
    sections = extract_sections(text)
    print("✂️ Extracted sections from text:")
    for k, v in sections.items():
        print(f"— {k}: {v[:100]}{'...' if len(v) > 100 else ''}")

    return predict_from_sections(sections)
