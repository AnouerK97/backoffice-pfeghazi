import pandas as pd
import os
import joblib

def load_model(module_name):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_directory, 'models', f'model-{module_name}.pkl')
    model = joblib.load(model_path)
    return model

def one_hot_encode_v1_0(df, categorical_columns):
    encoded_columns = []
    for column, categories in categorical_columns.items():
        encoded_columns.extend([f'{column}_{category}' for category in categories])

    df_encoded = pd.DataFrame(columns=encoded_columns, dtype=int)

    for column, categories in categorical_columns.items():
        for category in categories:
            df_encoded[f'{column}_{category}'] = (df[column] == category).astype(int)

    return df_encoded


def preprocess_data_v1_0(data, categorical_columns):
    df = pd.DataFrame(data, index=[0])

    for column in df.columns:
        if df[column].dtype != 'object':
            df[column] = df[column].astype(str)

    df_encoded = one_hot_encode_v1_0(df, categorical_columns)

    df = df.drop(columns=list(categorical_columns.keys()))

    df = pd.concat([df, df_encoded], axis=1)

    df = df.sort_index(axis=1)

    return df