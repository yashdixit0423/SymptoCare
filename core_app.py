import json
import streamlit as st

from core_predictor import predict
from core_report_generator import generate_report
from core_chatbot import medical_chat


# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="SymptoCare",
    page_icon="\U0001fa7a",
    layout="wide"
)


# -------------------------------------------------------
# Session State
# -------------------------------------------------------

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

if "prediction_context" not in st.session_state:
    st.session_state.prediction_context = None

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "report" not in st.session_state:
    st.session_state.report = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# -------------------------------------------------------
# Load Symptom Categories
# -------------------------------------------------------

with open("models/symptom_category_map.json", "r") as f:
    symptom_category_map = json.load(f)

categories = {}

for symptom, category in symptom_category_map.items():
    categories.setdefault(category, []).append(symptom)


# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.markdown(
    """
    <div style="padding: 1.2rem 0 0.4rem 0;">
        <h1 style="margin-bottom:0;">\U0001fa7a SymptoCare</h1>
        <p style="font-size:1.05rem; color:#4A5A6A; margin-top:0.3rem;">
            AI-assisted symptom analysis and clinical information support
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:

    st.subheader("Overview")

    st.markdown(
        """
        Select the symptoms you are experiencing to receive a model-generated
        disease prediction, followed by an AI-generated clinical summary.
        You may then consult the assistant for additional information.
        """
    )

    st.divider()

    st.markdown(
        """
        \u26a0\ufe0f **Disclaimer**

        This application is provided for educational and informational
        purposes only. It does not constitute medical advice and must not
        be used as a substitute for consultation with a qualified
        healthcare professional.
        """
    )

st.divider()

# -------------------------------------------------------
# Symptom Selection
# -------------------------------------------------------

st.header("Symptom Assessment")

selected_symptoms = []

for category in sorted(categories.keys()):

    with st.expander(category):

        options = [
            symptom.replace("_", " ").title()
            for symptom in categories[category]
        ]

        display_to_original = dict(zip(options, categories[category]))

        selected_display = st.multiselect(
            f"Select applicable {category.lower()} symptoms",
            options,
            key=category
        )

        for symptom in selected_display:
            selected_symptoms.append(display_to_original[symptom])


st.divider()

# -------------------------------------------------------
# Prediction Button
# -------------------------------------------------------

if st.button("Generate Assessment", use_container_width=True):

    if len(selected_symptoms) == 0:

        st.warning("Please select at least one symptom to proceed with the assessment.")

    else:

        result = predict(selected_symptoms)

        with st.spinner("Generating clinical report..."):

            report = generate_report(result)

        st.session_state.prediction_result = result
        st.session_state.prediction_context = result
        st.session_state.report = report


# -------------------------------------------------------
# Display Prediction
# -------------------------------------------------------

if st.session_state.prediction_result is not None:

    result = st.session_state.prediction_result

    report = st.session_state.report

    st.success("Assessment complete")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Predicted Condition")

        st.success(result["prediction"])

    with col2:

        st.subheader("Model Confidence")

        st.progress(result["confidence"])

        st.write(f"**{result['confidence']*100:.2f}%**")

    st.divider()

    st.subheader("Differential Predictions")

    for disease in result["top_predictions"]:

        st.write(
            f"\u2022 **{disease['disease']}** ({disease['confidence']*100:.2f}%)"
        )

    st.divider()

    st.subheader("Clinical Report")

    st.markdown(report)

    st.divider()

    if st.button("Consult the Assistant", use_container_width=True):

        st.session_state.chat_started = True


# -------------------------------------------------------
# Direct Chat
# -------------------------------------------------------

st.divider()

st.header("Clinical Information Assistant")

st.write(
    "You may bypass the symptom assessment and consult the assistant directly "
    "with any medical question."
)

if (
    not st.session_state.chat_started
    and st.session_state.prediction_result is None
):

    if st.button("Begin Consultation", use_container_width=True):

        st.session_state.chat_started = True


# -------------------------------------------------------
# Chat Interface
# -------------------------------------------------------

if st.session_state.chat_started:

    st.subheader("Consultation")

    if len(st.session_state.messages) == 0:

        st.info(
            "No messages yet. Ask a question below, or try one of the "
            "examples to get started."
        )

        example_questions = [
            "What lifestyle changes can help manage this condition?",
            "What are common warning signs I should watch for?",
            "Are there any home remedies that may help?",
        ]

        cols = st.columns(len(example_questions))

        for col, question in zip(cols, example_questions):

            with col:

                if st.button(question, use_container_width=True):

                    st.session_state.pending_question = question

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    user_input = st.chat_input("Enter your medical question...")

    if not user_input and st.session_state.get("pending_question"):

        user_input = st.session_state.pending_question

        st.session_state.pending_question = None

    if user_input:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        with st.chat_message("user"):

            st.markdown(user_input)

        # Placeholder until RAG is added

        history = ""

        for msg in st.session_state.messages:
            history += f"{msg['role']}: {msg['content']}\n"

        with st.spinner("Analyzing query..."):

            response = medical_chat(
                question=user_input,
                prediction_context=st.session_state.prediction_context,
                history=history
        )

      

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        with st.chat_message("assistant"):

            st.markdown(response)