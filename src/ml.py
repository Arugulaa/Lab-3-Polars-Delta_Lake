import mlflow
import mlflow.sklearn
import polars as pl
import pandas as pd
from deltalake import DeltaTable
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import os
import logging


logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOLD_FEATURES_PATH = os.path.join(BASE_DIR, "delta", "gold", "features")

def run_ml():
    logger.info("=== ML: Обучение моделей ===")

    required_cols = [
        "IATA_Code_Marketing_Airline", "Origin", "Dest", "season", "route",
        "month", "DayOfWeek", "hour", "Distance", "DepDelay",
        "target_regression", "is_delayed"
    ]

    df = (
        pl.scan_delta(GOLD_FEATURES_PATH)
        .filter(pl.col("target_regression").is_not_null())
        .select(required_cols)
        .collect()
        .drop_nulls()
    )
    logger.info(f"Загружено строк: {df.shape[0]}")

    cat_cols = ["IATA_Code_Marketing_Airline", "Origin", "Dest", "season", "route"]
    for col in cat_cols:
        le = LabelEncoder()
        df = df.with_columns(
            pl.Series(le.fit_transform(df[col].to_list())).alias(col)
        )

    feature_cols = [
        "IATA_Code_Marketing_Airline", "Origin", "Dest", "season", "route",
        "month", "DayOfWeek", "hour", "Distance", "DepDelay"
    ]

    X = df.select(feature_cols).to_pandas()
    y_reg = df["target_regression"].to_pandas()
    y_clf = df["is_delayed"].to_pandas()

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    gold_version = DeltaTable(GOLD_FEATURES_PATH).version()
    logger.info(f"Версия Gold таблицы: {gold_version}")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Flight_Delay_Prediction")

    # --- РЕГРЕССИЯ ---
    reg_models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        ),
        "LightGBM_Regressor": lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, random_state=42
        )
    }

    for name, model in reg_models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_reg_train)
            preds = model.predict(X_test)

            mae  = mean_absolute_error(y_reg_test, preds)
            rmse = mean_squared_error(y_reg_test, preds) ** 0.5
            r2   = r2_score(y_reg_test, preds)

            mlflow.log_param("model_type", name)
            mlflow.log_param("gold_version", gold_version)
            mlflow.log_param("features", str(feature_cols))
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            mlflow.sklearn.log_model(model, "model")

            logger.info(f"[REG] {name} → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")

    # --- КЛАССИФИКАЦИЯ ---
    clf_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        ),
        "LightGBM_Classifier": lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, random_state=42
        )
    }

    for name, model in clf_models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_clf_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_clf_test, preds)
            f1  = f1_score(y_clf_test, preds)
            roc = roc_auc_score(y_clf_test, proba)

            mlflow.log_param("model_type", name)
            mlflow.log_param("gold_version", gold_version)
            mlflow.log_param("features", str(feature_cols))
            mlflow.log_metric("Accuracy", acc)
            mlflow.log_metric("F1", f1)
            mlflow.log_metric("ROC_AUC", roc)
            mlflow.sklearn.log_model(model, "model")

            logger.info(f"[CLF] {name} → Accuracy: {acc:.3f}, F1: {f1:.3f}, ROC-AUC: {roc:.3f}")

    # --- FEATURE IMPORTANCE ---
    rf_reg = reg_models["RandomForestRegressor"]
    rf_clf = clf_models["RandomForestClassifier"]

    with mlflow.start_run(run_name="Feature_Importance"):
        mlflow.log_param("gold_version", gold_version)

        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "reg_importance": rf_reg.feature_importances_,
            "clf_importance": rf_clf.feature_importances_
        }).sort_values("reg_importance", ascending=False)

        mlflow.log_text(importance_df.to_csv(index=False), "feature_importance.csv")

        for _, row in importance_df.iterrows():
            mlflow.log_metric(f"reg_imp_{row['feature']}", row["reg_importance"])
            mlflow.log_metric(f"clf_imp_{row['feature']}", row["clf_importance"])

        logger.info("Feature Importance (регрессия):")
        logger.info("\n" + importance_df[["feature", "reg_importance"]].to_string(index=False))

    logger.info("=== ML: Готово ===")

if __name__ == "__main__":
    run_ml()