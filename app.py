from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

model = tf.keras.models.load_model(
    "bank_marketing_nn.keras"
)

preprocessor = joblib.load(
    "preprocessor.pkl"
)

month_map = {
    "jan":1,"feb":2,"mar":3,"apr":4,
    "may":5,"jun":6,"jul":7,"aug":8,
    "sep":9,"oct":10,"nov":11,"dec":12
}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "age": int(request.form["age"]),
        "job": request.form["job"],
        "marital": request.form["marital"],
        "education": request.form["education"],
        "default": request.form["default"],
        "balance": float(request.form["balance"]),
        "housing": request.form["housing"],
        "loan": request.form["loan"],
        "contact": request.form["contact"],
        "day": int(request.form["day"]),
        "month": request.form["month"],
        "duration": int(request.form["duration"]),
        "campaign": int(request.form["campaign"]),
        "pdays": int(request.form["pdays"]),
        "previous": int(request.form["previous"]),
        "poutcome": request.form["poutcome"]
    }

    df = pd.DataFrame([data])

    df["log_balance"] = np.log1p(
        df["balance"].clip(lower=0)
    )

    df["has_debt"] = (
        ((df["housing"] == "yes") &
         (df["loan"] == "yes"))
    ).astype(int)

    df["risk_score"] = (
        (df["default"] == "yes").astype(int)*3 +
        (df["housing"] == "yes").astype(int) +
        (df["loan"] == "yes").astype(int)
    )

    df["duration_per_contact"] = (
        df["duration"] /
        df["campaign"].clip(lower=1)
    )

    df["prev_success"] = (
        df["poutcome"] == "success"
    ).astype(int)

    df["month_num"] = df["month"].map(month_map)

    X = preprocessor.transform(df)

    probability = model.predict(
        X,
        verbose=0
    )[0][0]

    prediction = "YES" if probability >= 0.5 else "NO"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=round(float(probability)*100,2)
    )

if __name__ == "__main__":
    app.run(debug=True)