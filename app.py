import streamlit as st
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# Load necessary models and data
vectorizer = joblib.load('tfidf_vectorizer (1).pkl')
naive_bayes_model = joblib.load('Naive Bayes.pkl')
df = pd.read_csv('final_dataset (1).csv')

def recommend_jobs(user_skills):
    """Recommend jobs using Naïve Bayes."""
    user_vectorized = vectorizer.transform([user_skills])
    predicted_index = naive_bayes_model.predict(user_vectorized)[0]
    unique_jobs = df['Job'].unique()
    predicted_job = unique_jobs[predicted_index]
    return predicted_job

def find_missing_skills(job, user_skills):
    """Find missing skills based on job requirements."""
    job_skills_list = df[df["Job"] == job]["Skills"].values
    if len(job_skills_list) == 0:
        return []
    job_skills = set(job_skills_list[0].split(", "))
    user_skills_set = set(user_skills.split(","))
    missing_skills = job_skills - user_skills_set
    return list(missing_skills)

# Streamlit UI
st.title("Job Recommendation System")
st.write("Enter your skills to get a job recommendation and missing skills.")

user_input = st.text_area("Enter your skills (comma-separated):")
if st.button("Recommend Job"):
    if user_input:
        recommended_job = recommend_jobs(user_input)
        missing_skills = find_missing_skills(recommended_job, user_input)
        st.success(f"Recommended Job: {recommended_job}")
        st.warning(f"Missing Skills for {recommended_job}: {', '.join(missing_skills) if missing_skills else 'None'}")
    else:
        st.error("Please enter your skills!")
