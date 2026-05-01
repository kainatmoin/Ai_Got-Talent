# ================================================
# streamlit_app.py — Emotion Classifier Web App
# ================================================

import streamlit as st
import numpy as np
import sys
import os

st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="🎭",
    layout="centered"
)

# ------------------------------------------------
# Load Model
# ------------------------------------------------
@st.cache_resource
def load_model():
    try:
        from model_wrapper import MyModel
        return MyModel(), None
    except Exception as e:
        return None, str(e)

model, error = load_model()

if error:
    st.error(f"Model load error: {error}")
    st.stop()

# ------------------------------------------------
# Emotion Config
# ------------------------------------------------
EMOTIONS = ['admiration', 'anger', 'disgust', 'fear',
            'hope', 'joy', 'love', 'pride', 'sadness']

EMOTION_EMOJIS = {
    'admiration': '🤩',
    'anger'     : '😡',
    'disgust'   : '🤢',
    'fear'      : '😨',
    'hope'      : '🌟',
    'joy'       : '😄',
    'love'      : '❤️',
    'pride'     : '💪',
    'sadness'   : '😢',
}

EMOTION_COLORS = {
    'admiration': '#FFD700',
    'anger'     : '#FF4444',
    'disgust'   : '#90EE90',
    'fear'      : '#9B59B6',
    'hope'      : '#00BFFF',
    'joy'       : '#FFA500',
    'love'      : '#FF69B4',
    'pride'     : '#4169E1',
    'sadness'   : '#708090',
}

# ------------------------------------------------
# UI
# ------------------------------------------------
st.title("🎭 Emotion Classifier")
st.markdown("#### Detect emotions in any tweet — AI Got Talent, DevDay 26")
st.markdown("---")

tweet = st.text_area(
    "Enter a tweet:",
    placeholder="e.g. I am so proud of what we achieved today!",
    height=120
)

col1, col2 = st.columns([1, 4])
with col1:
    predict_btn = st.button("🔍 Predict", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

st.markdown("---")

# ------------------------------------------------
# Predict
# ------------------------------------------------
if predict_btn:
    if not tweet.strip():
        st.warning("Please enter a tweet first!")
    else:
        with st.spinner("Analyzing emotions..."):
            try:
                preds = model.predict([tweet])
                pred_row = preds[0]
                detected = [EMOTIONS[i] for i in range(9) if pred_row[i] == 1]

                if detected:
                    st.success("Emotions detected!")
                    emojis = " ".join([EMOTION_EMOJIS[e] for e in detected])
                    st.markdown(
                        f"<h1 style='text-align:center'>{emojis}</h1>",
                        unsafe_allow_html=True
                    )

                    cols = st.columns(len(detected))
                    for i, emotion in enumerate(detected):
                        with cols[i]:
                            color = EMOTION_COLORS[emotion]
                            st.markdown(
                                f"""
                                <div style='
                                    background-color: {color}22;
                                    border: 2px solid {color};
                                    border-radius: 12px;
                                    padding: 12px;
                                    text-align: center;
                                '>
                                    <div style='font-size:28px'>
                                        {EMOTION_EMOJIS[emotion]}
                                    </div>
                                    <div style='
                                        font-weight:bold;
                                        color:{color};
                                        font-size:14px;
                                        margin-top:4px;
                                    '>
                                        {emotion.upper()}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    st.markdown("---")
                    st.markdown("#### All Emotions Status")
                    cols2 = st.columns(9)
                    for i, emotion in enumerate(EMOTIONS):
                        with cols2[i]:
                            active  = emotion in detected
                            color   = EMOTION_COLORS[emotion] if active else "#cccccc"
                            opacity = "1" if active else "0.3"
                            st.markdown(
                                f"""
                                <div style='text-align:center; opacity:{opacity}'>
                                    <div style='font-size:20px'>
                                        {EMOTION_EMOJIS[emotion]}
                                    </div>
                                    <div style='font-size:10px; color:{color}; font-weight:bold'>
                                        {emotion}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.info("No strong emotion detected.")

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ------------------------------------------------
# Examples
# ------------------------------------------------
st.markdown("---")
st.markdown("#### Try these examples:")

examples = [
    "I am so proud of what we achieved together!",
    "This is frustrating and makes me really upset.",
    "Even though things are uncertain, I still have hope.",
    "I admire your courage but this is kind of scary.",
]

for example in examples:
    if st.button(f"💬 {example}", use_container_width=True):
        preds    = model.predict([example])
        detected = [EMOTIONS[i] for i in range(9) if preds[0][i] == 1]
        emojis   = " ".join([EMOTION_EMOJIS[e] for e in detected])
        st.markdown(f"**Tweet:** {example}")
        st.markdown(f"**Emotions:** {emojis} {', '.join(detected)}")

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:13px'>"
    "Built by Team Mindora • FAST NUCES Karachi • DevDay 26 🎮"
    "</p>",
    unsafe_allow_html=True
)