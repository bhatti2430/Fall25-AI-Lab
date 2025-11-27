from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

with open("Loan_Prediction_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
encoders = data["encoders"]
features = data["features"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    input_data = {
        "Gender": form.get("Gender"),
        "Married": form.get("Married"),
        "Education": form.get("Education"),
        "Self_Employed": form.get("Self_Employed"),
        "ApplicantIncome": float(form.get("ApplicantIncome")),
        "CoapplicantIncome": float(form.get("CoapplicantIncome")),
        "LoanAmount": float(form.get("LoanAmount")),
        "Loan_Amount_Term": float(form.get("Loan_Amount_Term")),
        "Credit_History": float(form.get("Credit_History")),
        "Property_Area": form.get("Property_Area"),
    }

    df = pd.DataFrame([input_data])

    # ---- FIX START ----
    for col, le in encoders.items():
        if col in df.columns:         # Avoid Loan_ID error
            df[col] = le.transform(df[col])
    # ---- FIX END ----

    df = df.reindex(columns=features, fill_value=0)

    scaled = scaler.transform(df)
    pred = model.predict(scaled)[0]

    result = "Approved" if pred == 1 else "Rejected"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
