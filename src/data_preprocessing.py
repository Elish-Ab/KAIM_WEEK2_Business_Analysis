import pandas as pd
from sqlalchemy import create_engine

def get_data_from_db(connection_string, query):
    """Fetch data from PostgreSQL database."""
    engine = create_engine(connection_string)
    return pd.read_sql(query, engine)

def handle_missing_values(df, column_name, method='mean'):
    """
    Handle missing values in the dataframe by replacing with mean, median, or mode.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - column_name: Column name (str) to fill missing values.
    - method: Method to replace missing values, either 'mean', 'median', or 'mode'. Default is 'mean'.
    
    Returns:
    - pandas DataFrame with missing values filled.
    """
    if method == 'mean':
        df[column_name].fillna(df[column_name].mean(), inplace=True)
    elif method == 'median':
        df[column_name].fillna(df[column_name].median(), inplace=True)
    elif method == 'mode':
        df[column_name].fillna(df[column_name].mode()[0], inplace=True)
    return df

def remove_outliers(df, column_name, threshold=3):
    """
    Remove outliers based on z-score (mean +/- threshold * std).
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - column_name: Column name (str) to check for outliers.
    - threshold: Number of standard deviations to use as the threshold for outliers.
    
    Returns:
    - pandas DataFrame without outliers.
    """
    mean = df[column_name].mean()
    std = df[column_name].std()
    upper_limit = mean + threshold * std
    lower_limit = mean - threshold * std
    df = df[(df[column_name] <= upper_limit) & (df[column_name] >= lower_limit)]
    return df
