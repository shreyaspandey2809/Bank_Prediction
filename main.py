import warnings
warnings.filterwarnings("ignore")

import os
import sqlite3
import threading
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

RANDOM_STATE   = 42
RETRAIN_EVERY  = 10          # auto-retrain after this many new feedback rows
DB_PATH        = "feedback.db"
MODEL_PATH     = "best_model.keras"
PREPROCESSOR_PATH = "preprocessor.pkl"
ORIGINAL_DATA  = "bank.csv"

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

MONTH_MAP = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
}

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                age           REAL, job TEXT, marital TEXT, education TEXT,
                "default"     TEXT, balance REAL, housing TEXT, loan TEXT,
                contact       TEXT, day REAL, month TEXT, duration REAL,
                campaign      REAL, pdays REAL, previous REAL, poutcome TEXT,
                predicted     INTEGER,
                probability   REAL,
                actual        INTEGER,
                used_in_train INTEGER DEFAULT 0,
                created_at    TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrain_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                triggered_at TEXT,
                rows_used    INTEGER,
                val_auc      REAL,
                val_accuracy REAL,
                epochs_run   INTEGER,
                status       TEXT
            )
        """)
        conn.commit()

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

init_db()

_retrain_lock   = threading.Lock()
_is_retraining  = False

def load_model_and_preprocessor():
    model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    pre   = joblib.load(PREPROCESSOR_PATH)         if os.path.exists(PREPROCESSOR_PATH) else None
    return model, pre

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_balance"]         = np.log1p(df["balance"].clip(lower=0))
    df["has_debt"]            = (
        (df["housing"] == "yes") & (df["loan"] == "yes")
    ).astype(int)
    df["risk_score"]          = (
        (df["default"] == "yes").astype(int) * 3 +
        (df["housing"] == "yes").astype(int) +
        (df["loan"]    == "yes").astype(int)
    )
    df["duration_per_contact"]= df["duration"] / df["campaign"].clip(lower=1)
    df["prev_success"]        = (df["poutcome"] == "success").astype(int)
    df["month_num"]           = df["month"].map(MONTH_MAP)
    return df

def retrain_model():
    global _is_retraining
    with _retrain_lock:
        if _is_retraining:
            return
        _is_retraining = True

    started = datetime.utcnow().isoformat()
    try:
        # 1. Load original bank.csv
        if not os.path.exists(ORIGINAL_DATA):
            raise FileNotFoundError(f"{ORIGINAL_DATA} not found next to main.py")

        df_orig = pd.read_csv(ORIGINAL_DATA).drop_duplicates()
        df_orig["deposit"] = df_orig["deposit"].map({"yes": 1, "no": 0})

        # 2. Pull all feedback rows from DB
        with get_db() as conn:
            rows = conn.execute(
                "SELECT age,job,marital,education,\"default\",balance,housing,loan,"
                "contact,day,month,duration,campaign,pdays,previous,poutcome,actual "
                "FROM feedback"
            ).fetchall()

        if rows:
            df_fb = pd.DataFrame([dict(r) for r in rows])
            df_fb = df_fb.rename(columns={"actual": "deposit"})
            df_combined = pd.concat([df_orig, df_fb], ignore_index=True)
        else:
            df_combined = df_orig

        df_combined = engineer_features(df_combined)

        TARGET = "deposit"
        X = df_combined.drop(columns=[TARGET])
        y = df_combined[TARGET]

        numerical_cols   = X.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
        )

        # 3. Rebuild preprocessor on combined data
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler())
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        preprocessor = ColumnTransformer([
            ("num", num_pipe, numerical_cols),
            ("cat", cat_pipe, categorical_cols)
        ])

        X_train_p = preprocessor.fit_transform(X_train)
        X_val_p   = preprocessor.transform(X_val)

        weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights = {0: weights[0], 1: weights[1]}

        # 4. Build fresh model
        input_dim = X_train_p.shape[1]
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(64,  activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(32,  activation="relu"),
            tf.keras.layers.Dense(1,   activation="sigmoid")
        ])
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"]
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, verbose=0
            )
        ]

        hist = model.fit(
            X_train_p, y_train,
            validation_data=(X_val_p, y_val),
            epochs=100, batch_size=64,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=0
        )

        # 5. Persist updated model + preprocessor
        model.save(MODEL_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)

        # 6. Mark all feedback rows as used
        with get_db() as conn:
            conn.execute("UPDATE feedback SET used_in_train = 1")
            conn.commit()

        best_auc = float(max(hist.history["val_auc"]))
        best_acc = float(max(hist.history["val_accuracy"]))
        epochs_run = len(hist.history["val_auc"])

        # 7. Log retrain event
        with get_db() as conn:
            conn.execute(
                "INSERT INTO retrain_log (triggered_at, rows_used, val_auc, val_accuracy, epochs_run, status) "
                "VALUES (?,?,?,?,?,?)",
                (started, len(df_combined), round(best_auc,4), round(best_acc,4), epochs_run, "success")
            )
            conn.commit()

        print(f"[retrain] Done — {len(df_combined)} rows | val_auc={best_auc:.4f} | epochs={epochs_run}")

    except Exception as e:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO retrain_log (triggered_at, rows_used, val_auc, val_accuracy, epochs_run, status) "
                "VALUES (?,?,?,?,?,?)",
                (started, 0, 0.0, 0.0, 0, f"error: {e}")
            )
            conn.commit()
        print(f"[retrain] ERROR: {e}")
    finally:
        _is_retraining = False

def maybe_trigger_retrain(background_tasks: BackgroundTasks):
    """Fire retrain in background if enough new feedback has accumulated."""
    with get_db() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE used_in_train = 0"
        ).fetchone()[0]
    if count >= RETRAIN_EVERY and not _is_retraining:
        background_tasks.add_task(retrain_model)

class CustomerInput(BaseModel):
    age:       float
    job:       str
    marital:   str
    education: str
    default:   str
    balance:   float
    housing:   str
    loan:      str
    contact:   str
    day:       float
    month:     str
    duration:  float
    campaign:  float
    pdays:     float
    previous:  float
    poutcome:  str

class FeedbackInput(BaseModel):
    age:        float
    job:        str
    marital:    str
    education:  str
    default:    str
    balance:    float
    housing:    str
    loan:       str
    contact:    str
    day:        float
    month:      str
    duration:   float
    campaign:   float
    pdays:      float
    previous:   float
    poutcome:   str
    predicted:  int
    probability:float
    actual:     int

app = FastAPI(title="Bank Deposit Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    model_ok = os.path.exists(MODEL_PATH)
    pre_ok   = os.path.exists(PREPROCESSOR_PATH)
    data_ok  = os.path.exists(ORIGINAL_DATA)
    with get_db() as conn:
        feedback_count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        pending        = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE used_in_train=0"
        ).fetchone()[0]
    return {
        "model_loaded": model_ok,
        "preprocessor_loaded": pre_ok,
        "bank_csv_found": data_ok,
        "feedback_total": feedback_count,
        "feedback_pending_retrain": pending,
        "retrain_threshold": RETRAIN_EVERY,
        "is_retraining": _is_retraining
    }

@app.post("/api/predict")
def predict(customer: CustomerInput):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise HTTPException(503, "Model or preprocessor not found. Run training first.")

    model, preprocessor = load_model_and_preprocessor()

    df = pd.DataFrame([customer.model_dump()])
    df = engineer_features(df)

    X = preprocessor.transform(df)
    prob = float(model.predict(X, verbose=0)[0][0])
    pred = int(prob >= 0.50)

    return {"prediction": pred, "probability": round(prob, 4)}

@app.post("/api/feedback")
def submit_feedback(fb: FeedbackInput, background_tasks: BackgroundTasks):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO feedback "
            "(age,job,marital,education,\"default\",balance,housing,loan,contact,"
            " day,month,duration,campaign,pdays,previous,poutcome,"
            " predicted,probability,actual,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                fb.age, fb.job, fb.marital, fb.education, fb.default,
                fb.balance, fb.housing, fb.loan, fb.contact,
                fb.day, fb.month, fb.duration, fb.campaign,
                fb.pdays, fb.previous, fb.poutcome,
                fb.predicted, fb.probability, fb.actual,
                datetime.utcnow().isoformat()
            )
        )
        conn.commit()
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE used_in_train=0"
        ).fetchone()[0]

    retrain_triggered = False
    if pending >= RETRAIN_EVERY and not _is_retraining:
        background_tasks.add_task(retrain_model)
        retrain_triggered = True

    return {
        "saved": True,
        "feedback_id": new_id,
        "pending_until_retrain": max(0, RETRAIN_EVERY - pending),
        "retrain_triggered": retrain_triggered
    }

@app.get("/api/stats")
def get_stats():
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        correct = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE predicted = actual"
        ).fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE used_in_train = 0"
        ).fetchone()[0]
        logs = conn.execute(
            "SELECT triggered_at, rows_used, val_auc, val_accuracy, epochs_run, status "
            "FROM retrain_log ORDER BY id DESC LIMIT 5"
        ).fetchall()

    accuracy = round(correct / total * 100, 1) if total > 0 else None
    return {
        "total_feedback": total,
        "correct_predictions": correct,
        "model_accuracy_pct": accuracy,
        "pending_feedback": pending,
        "retrain_threshold": RETRAIN_EVERY,
        "progress_to_retrain": min(pending, RETRAIN_EVERY),
        "is_retraining": _is_retraining,
        "retrain_history": [dict(r) for r in logs]
    }

@app.get("/api/feedback/history")
def feedback_history(limit: int = 50):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, age, job, marital, balance, duration, predicted, "
            "probability, actual, used_in_train, created_at "
            "FROM feedback ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

# Serve the frontend
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "Static UI not found. Place index.html in ./static/"}