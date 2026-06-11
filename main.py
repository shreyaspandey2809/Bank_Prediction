import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,classification_report)
from sklearn.utils.class_weight import compute_class_weight

DATA_PATH = "bank.csv"
TARGET = "deposit"

RANDOM_STATE = 42
TEST_SIZE = 0.20

BATCH_SIZE = 64
EPOCHS = 100

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset Shape: {df.shape}")

df = df.drop_duplicates()

df[TARGET] = df[TARGET].map({"yes": 1,"no": 0})

print("Feature Engineering...")

df["log_balance"] = np.log1p(df["balance"].clip(lower=0))

df["has_debt"] = (((df["housing"] == "yes") & (df["loan"] == "yes"))).astype(int)

df["risk_score"] = ((df["default"] == "yes").astype(int) * 3 + (df["housing"] == "yes").astype(int) + (df["loan"] == "yes").astype(int))

df["duration_per_contact"] = (df["duration"] / df["campaign"].clip(lower=1))

df["prev_success"] = (df["poutcome"] == "success").astype(int)

month_map = {
    "jan":1,"feb":2,"mar":3,"apr":4,
    "may":5,"jun":6,"jul":7,"aug":8,
    "sep":9,"oct":10,"nov":11,"dec":12
}

df["month_num"] = df["month"].map(month_map)

X = df.drop(columns=[TARGET])
y = df[TARGET]

numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print(f"Numerical Features: {len(numerical_cols)}")
print(f"Categorical Features: {len(categorical_cols)}")

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([("num", numeric_pipeline, numerical_cols),("cat", categorical_pipeline, categorical_cols)])

print("\nPreprocessing data...")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Processed Shape:", X_train_processed.shape)

weights = compute_class_weight(class_weight="balanced",classes=np.unique(y_train),y=y_train)

class_weights = { 0: weights[0], 1: weights[1]}

print("Class Weights:", class_weights)

input_dim = X_train_processed.shape[1]

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(input_dim,)),

    tf.keras.layers.Dense(256,activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(32,activation="relu"),

    tf.keras.layers.Dense(1,activation="sigmoid")

])

optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(name="auc"),
        "accuracy"
    ]
)

model.summary()

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=10,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True
    )

]

print("\nTraining Neural Network...\n")

history = model.fit(

    X_train_processed,
    y_train,

    validation_split=0.20,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    class_weight=class_weights,

    verbose=1
)

print("\nEvaluating Model...\n")

y_prob = model.predict(
    X_test_processed,
    verbose=0
).flatten()

y_pred = (y_prob >= 0.50).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("="*50)
print("FINAL RESULTS")
print("="*50)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nSaving model...")

model.save("bank_marketing_nn.keras")

joblib.dump(preprocessor,"preprocessor.pkl")

print("Model Saved Successfully!")

def predict_customer(customer_dict):

    model = tf.keras.models.load_model("bank_marketing_nn.keras")

    preprocessor = joblib.load("preprocessor.pkl")

    customer_df = pd.DataFrame([customer_dict])

    customer_df["log_balance"] = np.log1p(
        customer_df["balance"].clip(lower=0)
    )

    customer_df["has_debt"] = (
        (
            (customer_df["housing"] == "yes") &
            (customer_df["loan"] == "yes")
        )
    ).astype(int)

    customer_df["risk_score"] = (
        (customer_df["default"] == "yes").astype(int)*3 +
        (customer_df["housing"] == "yes").astype(int) +
        (customer_df["loan"] == "yes").astype(int)
    )

    customer_df["duration_per_contact"] = (customer_df["duration"] / customer_df["campaign"].clip(lower=1))

    customer_df["prev_success"] = (customer_df["poutcome"] == "success").astype(int)

    customer_df["month_num"] = (customer_df["month"].map(month_map))

    X_new = preprocessor.transform(customer_df)

    probability = model.predict(X_new,verbose=0)[0][0]

    prediction = int(probability >= 0.50)

    return {
        "prediction": prediction,
        "probability": round(float(probability), 4)
    }