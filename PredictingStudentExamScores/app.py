
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Student Score Predictor", layout="wide")

# LOAD MODEL
model = pickle.load(open("model.pkl", "rb"))

# TITLE
st.title("Student Exam Score Predictor")
st.markdown("### Predict student performance using ML")

st.markdown("---")

# INPUT SECTION
st.subheader("Enter Student Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 15, 30, 20)
    study_hours = st.slider("Study Hours/Day", 0.0, 10.0, 3.0)
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    exercise = st.slider("Exercise Frequency (0 - 7) days/week", 0, 7, 3)

with col2:
    social_media = st.slider("Social Media Hours", 0.0, 10.0, 2.0)
    netflix = st.slider("Netflix Hours", 0.0, 10.0, 1.0)
    mental_health = st.slider("Mental Health Rating", 1, 10, 8)

with col3:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    part_time = st.selectbox("Part Time Job", ["No", "Yes"])
    extra = st.selectbox("Extracurricular", ["No", "Yes"])

st.markdown("---")

# LIFESTYLE SECTION
st.subheader("Lifestyle & Background")

col4, col5, col6 = st.columns(3)

with col4:
    diet = st.selectbox("Diet Quality", ["Poor", "Fair", "Good"])

with col5:
    education = st.selectbox("Parental Education", ["High School", "Bachelor", "Master"])

with col6:
    internet = st.selectbox("Internet Quality", ["Poor", "Average", "Good"])

st.markdown("---")

# PREDICTION
if st.button("Predict Score"):

    input_data = pd.DataFrame({
        'age': [age],
        'study_hours_per_day': [study_hours],
        'social_media_hours': [social_media],
        'netflix_hours': [netflix],
        'sleep_hours': [sleep],
        'exercise_frequency': [exercise],
        'mental_health_rating': [mental_health],
        'gender': [gender],
        'part_time_job': [part_time],
        'extracurricular_participation': [extra],
        'diet_quality': [diet],
        'parental_education_level': [education],
        'internet_quality': [internet]
    })

    prediction = model.predict(input_data)[0]

    # RESULT DISPLAY
    st.markdown("Prediction Result")

    if prediction >= 75:
        st.success(f"Excellent! Predicted Score: {round(prediction, 2)}")
    elif prediction >= 50:
        st.warning(f"Average Performance: {round(prediction, 2)}")
    else:
        st.error(f"Needs Improvement: {round(prediction, 2)}")
