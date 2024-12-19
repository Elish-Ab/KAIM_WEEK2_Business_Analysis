import pandas as pd

def aggregate_user_behavior(df):
    """
    Aggregate user behavior data, e.g., session counts, avg session time, etc.
    
    Parameters:
    - df: pandas DataFrame containing user behavior data.
    
    Returns:
    - Aggregated DataFrame by user.
    """
    user_behavior = df.groupby('user_id').agg(
        total_sessions=('session_id', 'count'),
        avg_session_duration=('session_duration', 'mean'),
        total_spend=('spend_amount', 'sum')
    ).reset_index()
    return user_behavior

def aggregate_user_engagement(df):
    """
    Aggregate user engagement data such as active days and engagement score.
    
    Parameters:
    - df: pandas DataFrame containing user engagement data.
    
    Returns:
    - Aggregated DataFrame by user.
    """
    user_engagement = df.groupby('user_id').agg(
        active_days=('date', 'nunique'),
        engagement_score=('engagement', 'mean')
    ).reset_index()
    return user_engagement
