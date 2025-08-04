import streamlit as st
import os
import tempfile
from src.inference import run_inference_from_text, run_inference_from_file

# === Page setup ===
st.set_page_config(page_title="Resume Personality Predictor", page_icon="🔍")
st.title("🔍 Resume Personality Predictor")
st.markdown("Predict MBTI and Belbin personality types from a resume (PDF only or manual input).")

# === Input method toggle ===
method = st.radio("Choose input method:", ["💅 Upload PDF file", "✍️ Manual input"])

# === File upload mode ===
if method == "💅 Upload PDF file":
    uploaded_file = st.file_uploader("Upload your resume (.pdf only)", type=["pdf"])

    if uploaded_file is not None:
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name

        try:
            with st.spinner("🔄 Processing PDF..."):
                prediction = run_inference_from_file(temp_path)
            st.success(f"✅ **MBTI**: {prediction['mbti']}  \n✅ **Belbin**: {prediction['belbin']}")
        except Exception as e:
            st.error(f"❌ Error processing file:\n\n{e}")
        finally:
            os.remove(temp_path)

# === Manual input mode ===
else:
    st.markdown("### ✍️ Enter Resume Sections Manually")
    position = st.text_input("Desired Position")
    soft_skills = st.text_area("Soft Skills")
    summary = st.text_area("Summary / About")
    projects = st.text_area("Projects / Experience")

    if st.button("Predict"):
        try:
            with st.spinner("Predicting personality..."):
                prediction = run_inference_from_text(position, soft_skills, summary, projects)
            st.success(f"✅ **MBTI**: {prediction['mbti']}  \n✅ **Belbin**: {prediction['belbin']}")
        except Exception as e:
            st.error(f"❌ Error during prediction:\n\n{e}")

# === Personality type explanations ===
with st.expander("🧠 What do the MBTI groups mean?"):
    st.markdown("""
**MBTI Groups (based on 4 dichotomies):**

- **Analysts (NT):** Intuitive + Thinking  
  Logical, strategic, independent thinkers. Often analysts, architects, and visionaries.
  
- **Diplomats (NF):** Intuitive + Feeling  
  Empathetic, insightful, idealistic. Strong in communication and emotional intelligence.
  
- **Sentinels (SJ):** Sensing + Judging  
  Practical, responsible, organized. Excel at structure, planning, and dependability.
  
- **Explorers (SP):** Sensing + Perceiving  
  Flexible, hands-on, spontaneous. Thrive in dynamic environments and quick thinking.
""")

with st.expander("🧩 What are Belbin Team Roles?"):
    st.markdown("""
**Belbin Team Roles (behavioral types in teams):**

- **Plant:** Creative innovator, solves hard problems  
- **Resource Investigator:** Explores opportunities, communicates well  
- **Coordinator:** Mature, confident, delegates effectively  
- **Shaper:** Dynamic, pushes forward under pressure  
- **Monitor Evaluator:** Logical, impartial, strategic  
- **Teamworker:** Cooperative, diplomatic, builds harmony  
- **Implementer:** Practical, turns ideas into action  
- **Completer Finisher:** Detail-focused, meets deadlines  
- **Specialist:** Deep expertise in a specific area
""")