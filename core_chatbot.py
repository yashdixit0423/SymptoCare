from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from core_prompts import CHAT_PROMPT
from core_retriever import retrieve_context

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

prompt = PromptTemplate.from_template(CHAT_PROMPT)


def medical_chat(question, prediction_context=None, history=""):

    context = retrieve_context(question)

    if prediction_context:

        prediction = (
            f"Predicted Disease: {prediction_context['prediction']}\n"
            f"Confidence: {prediction_context['confidence']*100:.2f}%\n"
            f"Symptoms: {', '.join(prediction_context['selected_symptoms'])}"
        )

    else:

        prediction = "No prediction available."

    chain = prompt | llm

    response = chain.invoke(
        {
            "prediction_context": prediction,
            "context": context,
            "history": history,
            "question": question
        }
    )

    return response.content