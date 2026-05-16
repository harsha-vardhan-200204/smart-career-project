from flask import Flask, render_template, request
import pandas as pd
import joblib

# Create Flask app
app = Flask(__name__)


# Load trained model
model = joblib.load(
    "career_guidance_model.pkl"
)

# Load encoders
education_encoder = joblib.load("education_encoder.pkl")


skills_encoder = joblib.load("skills_encoder.pkl")


interests_encoder = joblib.load(
    "interests_encoder.pkl"
)

career_encoder = joblib.load(
    "career_encoder.pkl"
)


# Home Page
@app.route("/")
def home():
    return render_template(
        "index.html"
    )


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get input
        age = float(
            request.form["age"]
        )

        education_input = request.form[
            "education"
        ].strip()

        skills_input = request.form[
            "skills"
        ].strip().replace("\t", "")

        interests_input = request.form[
            "interests"
        ].strip()

        score = float(
            request.form["score"]
        )


        # Encode input values
        education = education_encoder.transform([
            education_input
        ])[0]

        skills = skills_encoder.transform([
            skills_input
        ])[0]

        interests = interests_encoder.transform([
            interests_input
        ])[0]


        # Create dataframe
        input_data = pd.DataFrame([[
            age,
            education,
            skills,
            interests,
            score
        ]])


        # Predict
        prediction = model.predict(
            input_data
        )


        # Convert prediction back to job name
        career_name = career_encoder.inverse_transform(
            prediction
        )[0]


        return render_template(
            "index.html",
            prediction_text=f"Predicted Career: {career_name}"
        )


    except Exception as e:

        return render_template(
            "index.html",
            prediction_text=f"Input Error: {str(e)}"
        )


# Run Flask
if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )