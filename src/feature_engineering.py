import pandas as pd
import numpy as np

def create_session_features(df):
    """
    Create features from session data (e.g., session length, time of day).
    
    Parameters:
    - df: pandas DataFrame containing session data.
    
    Returns:
    - pandas DataFrame with new session-based features.
    """
    df['session_start_hour'] = df['session_start_time'].dt.hour
    df['session_duration'] = (df['session_end_time'] - df['session_start_time']).dt.total_seconds()
    
    # Example: log transform of session duration
    df['log_session_duration'] = np.log1p(df['session_duration'])
    
    return df

def create_user_features(df):
    """
    Create user-level features (e.g., frequency of sessions, spending patterns).
    
    Parameters:
    - df: pandas DataFrame containing user data.
    
    Returns:
    - pandas DataFrame with new user-level features.
    """
    df['session_frequency'] = df.groupby('user_id')['session_id'].transform('count')
    df['avg_spend_per_session'] = df['total_spend'] / df['session_frequency']
    return df
