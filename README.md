# SymptoCare : AI-Powered Symptom Disease Assistant

An intelligent healthcare assistant that combines **Machine Learning**, **Large Language Models (LLMs)**, and **Retrieval-Augmented Generation (RAG)** to provide disease prediction, AI-generated medical reports, and conversational medical assistance.

The application predicts possible diseases from user-selected symptoms, explains the prediction using an LLM, and allows users to continue the conversation through a context-aware clinical information assistant powered by a medical knowledge base.

> **Disclaimer:** This application is developed for educational purposes only. It is not intended to diagnose, treat, or replace professional medical advice.

---

# Motivation

Traditional disease prediction models provide only a predicted label and confidence score, leaving users without context or explanation.

This project bridges that gap by combining machine learning with modern language models to deliver:

- Disease prediction
- AI-generated medical explanations
- Interactive medical Q&A
- Knowledge-grounded responses using Retrieval-Augmented Generation (RAG)

---

# Key Features

## Symptom Assessment

- Predict diseases from symptoms using a trained Logistic Regression model.
- Symptoms are organized into eight medical categories:
  - General
  - Skin
  - Gastrointestinal
  - Respiratory
  - Musculoskeletal
  - Urinary
  - Eye
  - Reproductive
- Displays:
  - Predicted condition
  - Model confidence
  - Differential predictions (top three most probable conditions)

---

## Clinical Report

After the assessment, the application generates a structured report using Groq's Llama model through LangChain.

The report includes:

- Disease overview
- Interpretation of prediction confidence
- Possible causes
- Home care recommendations
- When to seek medical attention
- Emergency warning signs
- Medical disclaimer

---

## Clinical Information Assistant

Users can interact with the assistant in two ways.

### Consult After Assessment

The assistant receives:

- Selected symptoms
- Predicted condition
- Confidence score

This enables follow-up questions based on the assessment.

### Direct Consultation

Users can bypass the symptom assessment and directly ask medical questions. Example question chips are shown to help users get started.

---

## Retrieval-Augmented Generation (RAG)

Instead of relying solely on an LLM's internal knowledge, the assistant retrieves relevant medical information from a local medical knowledge base.

The retrieval pipeline consists of:

Medical PDFs → Text Chunking → Sentence Embeddings → FAISS Vector Search → LangChain → Groq LLM

This helps generate more relevant and grounded responses.

---

# System Architecture

```
                    User
                      │
                      ▼
          Select Symptoms (Optional)
                      │
                      ▼
      Binary Feature Vector Creation
                      │
                      ▼
     Logistic Regression Classifier
                      │
                      ▼
 Disease Prediction + Confidence Score
                      │
                      ▼
     Clinical Report (LangChain + Groq)
                      │
                      ▼
       Consult the Assistant (Optional)
                      │
                      ▼
               User Question
                      │
                      ▼
      Retrieve Relevant Medical Chunks
               from FAISS Index
                      │
                      ▼
    Combine Retrieved Context + ML Prediction
                      │
                      ▼
           Groq Llama 3.3 70B LLM
                      │
                      ▼
        Clinical Information Assistant
```

---

# Technology Stack

| Category | Technologies |
|----------|--------------|
| Frontend | Streamlit |
| Machine Learning | Scikit-learn, Logistic Regression |
| LLM | Groq Llama 3.3 70B |
| Framework | LangChain |
| Vector Database | FAISS |
| Embeddings | HuggingFace Sentence Transformers |
| Document Processing | PyPDF |
| Programming Language | Python |

---

# Folder Structure

```
SymptoCare/

├── core_app.py
├── core_predictor.py
├── core_chatbot.py
├── core_retriever.py
├── core_report_generator.py
├── core_prompts.py
├── core_ingest.py
│
├── models/
│   ├── logistic_regression_model.joblib
│   ├── label_encoder.joblib
│   ├── disease_label_mapping.json
│   ├── symptom_category_map.json
│   └── feature_order.json
│
├── knowledge_base/
│   └── Medical PDF Documents
│
├── vectorstore/
│   ├── index.faiss
│   └── index.pkl
│
├── data/
│   └── symptom_disease_clean.csv
│
├── requirements.txt
├── README.md
└── .env
```

---

# Installation

## Clone the Repository

```bash
git clone <repository-url>
cd SymptoCare
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

# Build the Knowledge Base

Place your medical PDF files inside:

```
knowledge_base/
```

Generate the vector database:

```bash
python core_ingest.py
```

---

# Run the Application

```bash
streamlit run core_app.py
```

---

# Application Workflow

### Symptom Assessment

1. User selects symptoms.
2. Symptoms are converted into a binary feature vector.
3. Logistic Regression predicts the most probable condition.
4. Confidence score and differential predictions are displayed.
5. LangChain sends the prediction to Groq to generate a detailed clinical report.

### Clinical Information Assistant

Users can either:

- Consult the assistant after completing an assessment, or
- Begin a new consultation directly, with example question chips to help get started.

The assistant retrieves relevant information from the FAISS vector database before generating each response.

---

# Future Scope

Potential enhancements include:

- Voice-based symptom input
- Multi-language support
- Medical image analysis
- Electronic Health Record (EHR) integration
- Hospital and doctor recommendation system
- Appointment scheduling
- User authentication
- Conversation memory
- Fine-tuned medical language models

---

# Project Highlights

- Machine Learning-based disease prediction
- Retrieval-Augmented Generation (RAG)
- LangChain workflow
- AI-generated medical reports
- Interactive conversational assistant
- Streamlit web interface
- Modular architecture

---

# Author

**Shreya Tyagi**

B.Tech Computer Science & Engineering (Artificial Intelligence & Machine Learning)

---

# License

This project is developed for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis or treatment.
