import os

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from core_prompts import REPORT_PROMPT

load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2
)


prompt = PromptTemplate.from_template(REPORT_PROMPT)


def generate_report(prediction_result):

    chain = prompt | llm

    report = chain.invoke({

        "prediction":
            prediction_result["prediction"],

        "confidence":
            round(prediction_result["confidence"] * 100, 2),

        "symptoms":
            ", ".join(prediction_result["selected_symptoms"]),

        "top_predictions":
            ", ".join(
                [
                    f"{d['disease']} ({d['confidence']*100:.1f}%)"
                    for d in prediction_result["top_predictions"]
                ]
            )
    })

    return report.content