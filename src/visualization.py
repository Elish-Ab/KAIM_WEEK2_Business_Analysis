import matplotlib.pyplot as plt
import seaborn as sns

def plot_user_behavior(user_behavior_df):
    """
    Plot user behavior metrics such as total sessions, avg session duration, etc.
    
    Parameters:
    - user_behavior_df: pandas DataFrame containing aggregated user behavior data.
    
    Returns:
    - Matplotlib figure showing the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='user_id', y='total_sessions', data=user_behavior_df)
    plt.title('Total Sessions per User')
    plt.xlabel('User ID')
    plt.ylabel('Total Sessions')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_user_engagement(user_engagement_df):
    """
    Plot user engagement metrics.
    
    Parameters:
    - user_engagement_df: pandas DataFrame containing user engagement data.
    
    Returns:
    - Matplotlib figure showing the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='user_id', y='engagement_score', data=user_engagement_df)
    plt.title('Engagement Score per User')
    plt.xlabel('User ID')
    plt.ylabel('Engagement Score')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
