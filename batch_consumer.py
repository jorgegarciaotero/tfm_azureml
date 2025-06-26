#!/usr/bin/env python
# coding: utf-8

# In[1]:
import argparse
# Basics
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import random
import plotly.express as px
import joblib
from datetime import timedelta
from typing import List, Tuple, Dict,Optional
from tensorflow.keras.models import load_model

# Azure
from adlfs import AzureBlobFileSystem
from typing import Tuple, List, Optional
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Models
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler,LabelEncoder
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc ,  precision_score, recall_score
from tensorflow.keras.metrics import AUC, Precision, Recall

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall
from keras_tuner import RandomSearch

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from tensorflow.keras.models import load_model



def load_data_from_dl(account_name: str,container_name: str,relative_path: str,access_key: str)->pd.DataFrame:
    """
    Loads all Parquet files from an Azure Blob Storage path into a single DataFrame.
    Args:
        - account_name (str): Azure Storage account name.
        - container_name (str): Name of the container.
        - relative_path (str): Path inside the container to search for .parquet files.
        - access_key (str): Storage account access key.
    Returns:
        - df (pd.DataFrame): Combined DataFrame from all found Parquet files.
    Raises:
        - ValueError: If no Parquet files are found in the path.
    """
    abfs = AzureBlobFileSystem(account_name=account_name, account_key=access_key)


    all_files = abfs.glob(f"{container_name}/{relative_path}/*.parquet")
    print(f"folder: {all_files}")

    if not all_files:
        raise ValueError("Not found .parquet files")

    dfs = []
    for f in all_files:
        print(f"Reading files: {f}")
        with abfs.open(f, "rb") as fp:
            dfs.append(pd.read_parquet(fp))

    df = pd.concat(dfs, ignore_index=True)
    print(df.head())
    return df,abfs




def prepare_data(
    df: pd.DataFrame,
    targets: list
):
    """
    Prepares a DataFrame for LSTM modeling: imputes missing values and applies MinMax scaling.

    Args:
        df (pd.DataFrame): Input data with features + target + symbol + date.
        targets (list): List of target column names.

    Returns:
        pd.DataFrame: Scaled dataframe with symbol and date preserved.
        MinMaxScaler: The fitted scaler object.
    """
    df_clean = df.copy()

    # Separar columnas a preservar
    symbol_col = df_clean["symbol"]
    date_col = df_clean["date"]

    # Eliminar columnas no necesarias para el modelo
    df_clean = df_clean.drop(columns=["symbol", "date"], errors="ignore")

    # Codificación de categóricas si existieran (seguridad)
    for col in df_clean.select_dtypes(include=["object", "category"]).columns:
        df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))

    # Separar features y targets
    feature_cols = [col for col in df_clean.columns if col not in targets]
    X = df_clean[feature_cols]
    y = df_clean[targets]

    # Imputación + escalado
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df_clean.index)
    df_scaled = pd.concat([df_scaled, y], axis=1)

    # Reincorporar columnas para split posterior
    df_scaled["symbol"] = symbol_col.values
    df_scaled["date"] = date_col.values

    return df_scaled, scaler



def model_evaluation(
    y_test: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series
) -> Tuple[float, float, float, np.ndarray, float, float]:
    """
    Evaluates the classification model and plots metrics.

    Args:
        y_test (pd.Series): True target values.
        y_pred (pd.Series): Predicted class values.
        y_prob (pd.Series): Predicted probabilities for class 1.

    Returns:
        Tuple containing Accuracy, F1 Score, ROC AUC, Confusion Matrix, Precision, Recall.
    """
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'])
    ax.set_yticklabels(['No', 'Yes'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return acc, f1, roc, cm, precision, recall



def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
  """
  Cleans a DataFrame by dropping unnecessary columns and handling missing values.

  Args:
      df (pd.DataFrame): Input DataFrame.

  Returns:
      pd.DataFrame: Cleaned DataFrame.
  """
  columns_to_drop = [
      'capital_gains',
      'ret_next_3m', 'ret_next_6m', 'ret_next_1y',
      'price_lead_3m', 'price_lead_6m', 'price_lead_1y',
      'open_v', 'high', 'low', 'dividends', 'stock_splits',
      'is_dividend_day', 'is_stock_split', 'gap_open', 'price_range',
      'tr_1', 'tr_2', 'tr_3', 'sma_5', 'bollinger_upper',
      'bollinger_lower', 'ema_12', 'macd_line'
  ]


  print(f"Shape before: {df.shape}")
  df = df.drop(columns=columns_to_drop, errors='ignore').copy()
  numeric_cols = df.select_dtypes(include=["float64", "int64", "int32"]).columns
  imputer = SimpleImputer(strategy="mean")
  df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
  print(f"Shape after: {df.shape}")
  return df;




def build_and_split_sequences_by_symbol(
    df: pd.DataFrame,
    target_column: str,
    sequence_length: int = 60,
    test_size: float = 0.2
):
    """
    Builds sequential data for each symbol independently and performs temporal train-test split.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame with features and target.
        target_column (str): Column name for binary target.
        sequence_length (int): Time window for each sequence.
        test_size (float): Proportion of test samples (applied per symbol).

    Returns:
        Tuple of numpy arrays: X_train, X_test, y_train, y_test
    """
    X_train, y_train, X_test, y_test = [], [], [], []

    symbols = df["symbol"].unique()
    for symbol in symbols:
        df_symbol = df[df["symbol"] == symbol].copy()
        df_symbol = df_symbol.sort_values("date")

        if len(df_symbol) <= sequence_length:
            continue  # skip if not enough data

        df_symbol[target_column] = df_symbol[target_column].astype(int)
        features = df_symbol.drop(columns=["date", "symbol", "target_3m", "target_6m", "target_1y"], errors="ignore")
        target = df_symbol[target_column].values

        split_idx = int(len(features) * (1 - test_size))
        for i in range(len(features) - sequence_length):
            if i + sequence_length >= len(features):
                continue  # avoid index error

            X_seq = features.iloc[i:i + sequence_length].values.astype(np.float32)
            y_val = target[i + sequence_length]

            if i + sequence_length < split_idx:
                X_train.append(X_seq)
                y_train.append(y_val)
            else:
                X_test.append(X_seq)
                y_test.append(y_val)

    return (
        np.array(X_train),
        np.array(X_test),
        np.array(y_train),
        np.array(y_test)
    )




def save_to_dl(df, file_name, target_date, container_name, account_name, access_key):
    """
    Saves the DataFrame as CSV in Azure Blob Storage.
    """
    abfs = AzureBlobFileSystem(account_name=account_name, account_key=access_key)
    full_path = f"{container_name}/predictions/{file_name}/{file_name}_{target_date}.csv"
    print(full_path)

    with abfs.open(full_path, "wb") as fp:
        df.to_csv(fp, index=False)






def predict_one_day(df_cut, model, scaler_minmax, target_column, seq_len, target_date):
    """
    Realiza predicciones LSTM para cada símbolo con exactamente seq_len días previos a target_date.
    """
    symbols = df_cut["symbol"].unique()
    df_last_seqs = []

    for symbol in symbols:
        df_sym = df_cut[df_cut["symbol"] == symbol].copy().sort_values("date")
        if len(df_sym) >= seq_len:
            df_seq = df_sym.tail(seq_len)
            if len(df_seq) == seq_len:
                df_last_seqs.append(df_seq)

    df_pred = pd.concat(df_last_seqs).sort_values(["symbol", "date"]).reset_index(drop=True)
    meta_cols = df_pred[["symbol", "date"]].copy()

    df_pred["date"] = pd.to_datetime(df_pred["date"], errors="coerce")
    df_pred["year"] = df_pred["date"].dt.year
    df_pred["month"] = df_pred["date"].dt.month
    df_pred["dayofweek"] = df_pred["date"].dt.dayofweek
    df_pred["symbol"] = LabelEncoder().fit_transform(df_pred["symbol"].astype(str))

    for col in df_pred.select_dtypes(include=["object", "category"]).columns:
        df_pred[col] = LabelEncoder().fit_transform(df_pred[col].astype(str))

    for t in ["target_3m", "target_6m", "target_1y"]:
        df_pred = df_pred.drop(columns=t, errors="ignore")

    feature_cols = [
        'symbol','close_v', 'volume', 'prev_close', 'prev_volume', 'daily_return',
        'close_change_pct', 'intraday_volatility', 'log_return', 'volume_change_pct',
        'sma_20', 'delta', 'gain', 'loss', 'rsi_14', 'rel_volume',
        'ema_26', 'macd_signal', 'macd_histogram', 'true_range', 'atr_14',
        'candle_body', 'upper_wick', 'lower_wick', 'candle_color', 'momentum_10',
        'roc_10', 'var_95', 'year', 'month', 'dayofweek'
    ]
    df_clean = df_pred[feature_cols]

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    df_clean[df_clean.columns] = imputer.fit_transform(df_clean)

    X_scaled = scaler_minmax.transform(df_clean.values)
    n_features = df_clean.shape[1]
    n_symbols = X_scaled.shape[0] // seq_len
    X_seq = X_scaled.astype("float32").reshape(n_symbols, seq_len, n_features)

    #y_prob = model.predict(X_seq, verbose=0).squeeze()
    import tensorflow as tf
    infer = model.signatures["serving_default"]
    print(infer.structured_input_signature)  # Verifica el nombre aquí
    inputs = tf.constant(X_seq)
    results = infer(input_layer=inputs)  # Cambia 'input_layer' por el nombre que imprimió la línea anterior
    y_prob = list(results.values())[0].numpy().squeeze()
    y_pred = (y_prob >= 0.5).astype(int)



    symbols_out = (
        meta_cols.groupby(np.arange(len(meta_cols)) // seq_len)
        .tail(1)["symbol"]
        .reset_index(drop=True)
    )

    return pd.DataFrame({
        "symbol": symbols_out,
        f"p_up_{target_column}": y_prob,
        "pred_up": y_pred,
        "calc_date": target_date
    }).sort_values(f"p_up_{target_column}", ascending=False)

def azureml_to_blob_path(azureml_path: str) -> str:
    return azureml_path.split('/paths/')[-1]

def main():
    # Parse CLI args  ───────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_date", type=str, required=True)
    args = parser.parse_args()

    TARGET_DATE = pd.Timestamp(args.target_date)

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    print(f"config: {config}")
    account_name = config['storage']['storage_account_name']
    container_name = config['storage']['container_name']
    relative_path = config['storage']['relative_path']
    access_key = config['storage']['access_key']

    df_full,fs = load_data_from_dl(account_name, container_name, relative_path, access_key)

    
    
    '''remote_model_path = "smart-wallet-dl/models/improved_lstm.keras"
    remote_scaler_path = "smart-wallet-dl/models/scaler_improved_lstm.pkl"
    fs.get(remote_model_path, "improved_lstm.keras")
    fs.get(remote_scaler_path, "scaler_improved_lstm.pkl")
    model_6m = load_model("improved_lstm.keras", compile=False)
    scaler_minmax_6m = joblib.load("scaler_improved_lstm.pkl")
    model_6m.export('improved_lstm_savedmodel')'''

   
    remote_model_path = "smart-wallet-dl/models/improved_lstm_savedmodel"
    remote_scaler_path = "smart-wallet-dl/models/scaler_improved_lstm.pkl"

    # Descargar carpeta modelo (si tu fs soporta descarga de directorios, si no, adapta)
    fs.get(remote_model_path, "improved_lstm_savedmodel", recursive=True)
    fs.get(remote_scaler_path, "scaler_improved_lstm.pkl")

    #model_6m = load_model("improved_lstm_savedmodel")
    from tensorflow import saved_model
    model_6m = saved_model.load("improved_lstm_savedmodel")
    scaler_minmax_6m = joblib.load("scaler_improved_lstm.pkl")

    print("Pasa!!!!!!!!!!!!!!!!")
    targets = ['target_3m', 'target_6m', 'target_1y']

    top_symbols = (
        df_full.groupby("symbol")
        .size()
        .sort_values(ascending=False)
        .head(500)
        .index
    )
    df_full = df_full[df_full["symbol"].isin(top_symbols)].copy()
    df_full["date"]=pd.to_datetime(df_full["date"])

    target_date = TARGET_DATE  
    seq_len = 60

    df_filtered = df_full[df_full["date"] < target_date].copy()

    latest_allowed = target_date - pd.Timedelta(days=1)
    valid_symbols = (
        df_filtered[df_filtered["date"] == latest_allowed]["symbol"]
        .value_counts()
        .loc[lambda x: x >= 1]
        .index
    )

    df_last = (
        df_filtered[df_filtered["symbol"].isin(valid_symbols)]
        .sort_values(["symbol", "date"])
        .groupby("symbol", group_keys=False)
        .tail(seq_len)
    )

    out_6m = predict_one_day(
        df_last,
        model_6m,
        scaler_minmax_6m,
        target_column="target_6m",
        seq_len=seq_len,
        target_date=target_date
        )

    out_6m.to_csv("out.csv",sep=";")

    target_path = target_date.strftime("%Y%m%d")  
    save_to_dl(out_6m, "6_months", target_path, container_name, account_name, access_key)



if __name__ == "__main__":
    main()

