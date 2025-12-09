import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64
import streamlit.components.v1 as components

st.set_page_config(page_title="Career Prediction System", page_icon="🥲")


def get_video_base64(video_path):
    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        b64 = base64.b64encode(video_bytes).decode()
        return f"data:video/mp4;base64,{b64}"
    except FileNotFoundError:
        return None


def show_error_page(error_detail):
    video_file_path = "snaptik.vn_Zrnji.mp4"

    video_src = get_video_base64(video_file_path)

    if video_src is None:
        video_src = "https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.mp4"

    error_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #1a1a1a; color: #fff; text-align: center; padding: 20px; }}
            h1 {{ color: #FF4B4B; font-size: 2.5em; }}
            video {{ max-width: 90%; max-height: 60vh; border-radius: 10px; margin: 20px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }}
            .detail {{ color: #ccc; font-size: 0.9em; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <h1>Well! The browser has been catted</h1>
        <h3>(System is down, enjoy the cat instead)</h3>

        <p class="detail">Error: {error_detail}</p>

        <video autoplay loop muted playsinline src="{video_src}">
            Your browser does not support the video tag.
        </video>
    </body>
    </html>
    """

    components.html(error_html, height=700)
    st.stop()

@st.cache_resource
def load_data():
    try:
        model = joblib.load('career_prediction_model.pkl')
        encoder = joblib.load('label_encoder.pkl')

        try:
            scaler = joblib.load('scaler.pkl')
        except:
            scaler = None

        return model, encoder, scaler
    except Exception as e:
        return None, None, str(e)

try:
    model, encoder, scaler = load_data()

    if model is None:
        st.error("Model files not found!")
        st.stop()

    st.title("Student Career Prediction System")
    st.write("Please assess yourself on a scale of **1 to 10** for the following skills and traits.")

    st.sidebar.header("Enter Your Scores (Scale 1-10)")

    if st.sidebar.button("Test Error Page"):
        show_error_page("Test Error Page =)))")

    user_input = {}

    st.sidebar.subheader("Technical Skills")
    tech_cols = ['Computer Architecture', 'Programming Skills', 'Project Management', 'Communication skills']

    for col in tech_cols:
        user_input[col] = st.sidebar.slider(f"{col}", 1, 10, 5)

    st.sidebar.subheader("Personality Traits")
    personality_cols = [
        'Openness', 'Conscientousness', 'Extraversion', 'Agreeableness',
        'Emotional_Range', 'Conversation', 'Openness to Change',
        'Hedonism', 'Self-enhancement', 'Self-transcendence'
    ]

    soft_skills_raw_values = []

    for col in personality_cols:
        val_1_to_10 = st.sidebar.slider(f"{col}", 1, 10, 5)

        soft_skills_raw_values.append(val_1_to_10)

        user_input[col] = val_1_to_10 / 10.0

    tech_score_model = np.mean([user_input[c] for c in tech_cols])
    soft_skill_score_model = np.mean([user_input[c] for c in personality_cols])

    user_input['tech_score'] = tech_score_model
    user_input['soft_skill_score'] = soft_skill_score_model

    tech_score_display = tech_score_model
    soft_skill_score_display = np.mean(soft_skills_raw_values)

    col1, col2 = st.columns(2)
    col1.metric("Technical Competency (Avg)", f"{tech_score_display:.1f} / 10")
    col2.metric("Soft Skills Competency (Avg)", f"{soft_skill_score_display:.1f} / 10")

    if st.button("Predict Career", type="primary"):
        input_df = pd.DataFrame([user_input])

        ordered_cols = tech_cols + personality_cols + ['tech_score', 'soft_skill_score']
        input_df = input_df[ordered_cols]

        if scaler is not None:
            if hasattr(scaler, 'feature_names_in_'):
                input_df = input_df[scaler.feature_names_in_]
            input_processed = scaler.transform(input_df)
        else:
            input_processed = input_df

        prediction_idx = model.predict(input_processed)[0]
        prediction_label = encoder.inverse_transform([prediction_idx])[0]

        st.success(f"Recommendation: You are best suited for the role of **{prediction_label}**!")

        if hasattr(model, "predict_proba"):
            proba = np.max(model.predict_proba(input_processed)) * 100
            st.info(f"AI Confidence Level: {proba:.1f}%")

except Exception as e:
    show_error_page(str(e))