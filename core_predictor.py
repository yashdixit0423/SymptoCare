import json
import joblib
import numpy as np


# ----------------------------
# Load all artifacts once
# ----------------------------

MODEL_PATH = "models/logistic_regression_model.joblib"
LABEL_ENCODER_PATH = "models/label_encoder.joblib"
DISEASE_MAPPING_PATH = "models/disease_label_mapping.json"
FEATURE_ORDER_PATH = "models/feature_order.json"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

with open(DISEASE_MAPPING_PATH, "r") as f:
    disease_mapping = json.load(f)

with open(FEATURE_ORDER_PATH, "r") as f:
    feature_order = json.load(f)


# ----------------------------
# Create binary feature vector
# ----------------------------

def create_feature_vector(selected_symptoms):

    feature_vector = []

    selected = set(selected_symptoms)

    for symptom in feature_order:

        if symptom in selected:
            feature_vector.append(1)

        else:
            feature_vector.append(0)

    return np.array(feature_vector).reshape(1, -1)


# ----------------------------
# Predict disease
# ----------------------------

def predict(selected_symptoms):

    features = create_feature_vector(selected_symptoms)

    predicted_class = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    confidence = float(np.max(probabilities))

    disease_name = disease_mapping[str(predicted_class)]

    top_indices = np.argsort(probabilities)[::-1][:3]

    top_predictions = []

    for idx in top_indices:

        top_predictions.append(
            {
                "disease": disease_mapping[str(idx)],
                "confidence": round(float(probabilities[idx]), 4)
            }
        )

    return {

        "prediction": disease_name,

        "confidence": round(confidence, 4),

        "selected_symptoms": selected_symptoms,

        "top_predictions": top_predictions
    }