import streamlit as st
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_md")

st.title("🧠 Cognitive Resume Intelligence Tool")
st.write("AI-based ATS parser ranking candidate-role fit.")

# Load data
data = pd.read_csv("resumes.csv")
jd_text = open("job_description.txt").read()
jd_doc = nlp(jd_text)

def compute_similarity(resume):
    doc = nlp(resume)
    return round(jd_doc.similarity(doc)*100, 2)

data["Fit_Score (%)"] = data["resume_text"].apply(compute_similarity)
ranked = data.sort_values("Fit_Score (%)", ascending=False)

st.dataframe(ranked)

best = ranked.iloc[0]
st.subheader("Top Match")
st.write(f"Candidate: **{best['name']}** | Fit Score: **{best['Fit_Score (%)']}%**")
