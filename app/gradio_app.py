import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import joblib
import pandas as pd
import gradio as gr
from pathlib import Path

from src.preprocessing.feature_engineering import create_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGRESSION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "regression" / "best_regression_model.pkl"
)

CLASSIFICATION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "classification" / "best_classification_model.pkl"
)

REGRESSION_PREPROCESSOR_PATH = (
    PROJECT_ROOT / "models" / "preprocessing" / "regression_preprocessor.pkl"
)

CLASSIFICATION_PREPROCESSOR_PATH = (
    PROJECT_ROOT / "models" / "preprocessing" / "classification_preprocessor.pkl"
)


regression_model = joblib.load(REGRESSION_MODEL_PATH)
classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)

regression_preprocessor = joblib.load(REGRESSION_PREPROCESSOR_PATH)
classification_preprocessor = joblib.load(CLASSIFICATION_PREPROCESSOR_PATH)


def create_input_dataframe(
    chol,
    stab_glu,
    hdl,
    ratio,
    location,
    age,
    gender,
    height,
    weight,
    frame,
    bp_1s,
    bp_1d,
    waist,
    hip,
    time_ppn
):
    return pd.DataFrame([{
        "chol": chol,
        "stab.glu": stab_glu,
        "hdl": hdl,
        "ratio": ratio,
        "location": location,
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "frame": frame,
        "bp.1s": bp_1s,
        "bp.1d": bp_1d,
        "waist": waist,
        "hip": hip,
        "time.ppn": time_ppn
    }])


def predict_glyhb(
    chol, stab_glu, hdl, ratio, location, age, gender,
    height, weight, frame, bp_1s, bp_1d, waist, hip, time_ppn
):
    data = create_input_dataframe(
        chol, stab_glu, hdl, ratio, location, age, gender,
        height, weight, frame, bp_1s, bp_1d, waist, hip, time_ppn
    )

    data = create_features(data)
    processed_data = regression_preprocessor.transform(data)

    prediction = regression_model.predict(processed_data)[0]

    return f"Predicted GlyHb: {prediction:.2f}%"


def predict_diabetes(
    chol, stab_glu, hdl, ratio, location, age, gender,
    height, weight, frame, bp_1s, bp_1d, waist, hip, time_ppn
):
    data = create_input_dataframe(
        chol, stab_glu, hdl, ratio, location, age, gender,
        height, weight, frame, bp_1s, bp_1d, waist, hip, time_ppn
    )

    processed_data = classification_preprocessor.transform(data)

    prediction = int(classification_model.predict(processed_data)[0])

    if prediction == 1:
        result = "Diabetes class: Positive (1)"
    else:
        result = "Diabetes class: Negative (0)"

    if hasattr(classification_model, "predict_proba"):
        probabilities = classification_model.predict_proba(processed_data)[0]
        probability = probabilities[list(classification_model.classes_).index(1)]
        result += f"\nModel-estimated probability of class 1: {probability:.2%}"

    return result


with gr.Blocks(title="Diabetes Prediction System") as demo:

    gr.Markdown(
        """
        # Diabetes Prediction System

        Predict GlyHb percentage using regression or predict the
        diabetes class using classification.
        
        Enter the patient's health measurements below.
        """
    )

    with gr.Tab("GlyHb Regression"):

        gr.Markdown("### Predict GlyHb percentage")

        regression_inputs = []

        with gr.Row():
            chol = gr.Number(label="Cholesterol (chol)")
            stab_glu = gr.Number(label="Stable Glucose (stab.glu)")
            hdl = gr.Number(label="HDL")
            ratio = gr.Number(label="Cholesterol Ratio")

        with gr.Row():
            location = gr.Textbox(label="Location")
            age = gr.Number(label="Age")
            gender = gr.Textbox(label="Gender")
            height = gr.Number(label="Height (inches)")

        with gr.Row():
            weight = gr.Number(label="Weight (pounds)")
            frame = gr.Textbox(label="Body Frame")
            bp_1s = gr.Number(label="Systolic Blood Pressure")
            bp_1d = gr.Number(label="Diastolic Blood Pressure")

        with gr.Row():
            waist = gr.Number(label="Waist")
            hip = gr.Number(label="Hip")
            time_ppn = gr.Number(label="Time After Meal")

        regression_button = gr.Button("Predict GlyHb", variant="primary")
        regression_output = gr.Textbox(label="Regression Result")

        regression_button.click(
            fn=predict_glyhb,
            inputs=[
                chol, stab_glu, hdl, ratio, location, age, gender,
                height, weight, frame, bp_1s, bp_1d, waist, hip, time_ppn
            ],
            outputs=regression_output
        )

    with gr.Tab("Diabetes Classification"):

        gr.Markdown("### Predict diabetes class")

        with gr.Row():
            c_chol = gr.Number(label="Cholesterol (chol)")
            c_stab_glu = gr.Number(label="Stable Glucose (stab.glu)")
            c_hdl = gr.Number(label="HDL")
            c_ratio = gr.Number(label="Cholesterol Ratio")

        with gr.Row():
            c_location = gr.Textbox(label="Location")
            c_age = gr.Number(label="Age")
            c_gender = gr.Textbox(label="Gender")
            c_height = gr.Number(label="Height (inches)")

        with gr.Row():
            c_weight = gr.Number(label="Weight (pounds)")
            c_frame = gr.Textbox(label="Body Frame")
            c_bp_1s = gr.Number(label="Systolic Blood Pressure")
            c_bp_1d = gr.Number(label="Diastolic Blood Pressure")

        with gr.Row():
            c_waist = gr.Number(label="Waist")
            c_hip = gr.Number(label="Hip")
            c_time_ppn = gr.Number(label="Time After Meal")

        classification_button = gr.Button(
            "Predict Diabetes Class",
            variant="primary"
        )

        classification_output = gr.Textbox(label="Classification Result")

        classification_button.click(
            fn=predict_diabetes,
            inputs=[
                c_chol, c_stab_glu, c_hdl, c_ratio, c_location,
                c_age, c_gender, c_height, c_weight, c_frame,
                c_bp_1s, c_bp_1d, c_waist, c_hip, c_time_ppn
            ],
            outputs=classification_output
        )


if __name__ == "__main__":
    demo.launch()
