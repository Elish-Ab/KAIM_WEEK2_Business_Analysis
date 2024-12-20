import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Analyzer:
    @staticmethod
    def get_top_handsets(df, n=10):
        """
        Identify the top N handsets used by the customers.
        """
        print(df.columns)
        if 'handset_type' not in df.columns:
            raise KeyError("Column 'handset_type' is missing from the DataFrame.")
        top_handsets = df['handset_type'].value_counts().head(n)
        return top_handsets

    @staticmethod
    def get_top_manufacturers(df, n=3):
        """
        Identify the top N handset manufacturers.
        """
        if 'handset_manufacturer' not in df.columns:
            raise KeyError("Column 'handset_manufacturer' is missing from the DataFrame.")
        top_manufacturers = df['handset_manufacturer'].value_counts().head(n)
        return top_manufacturers

    @staticmethod
    def get_top_handsets_per_manufacturer(df, top_manufacturers):
        """
        Identify the top 5 handsets for each of the top manufacturers.
        """
        if 'handset_manufacturer' not in df.columns or 'handset_type' not in df.columns:
            raise KeyError("Columns 'handset_manufacturer' or 'handset_type' are missing from the DataFrame.")
        results = {}
        for manufacturer in top_manufacturers.index:
            handsets = df[df['handset_manufacturer'] == manufacturer]['handset_type'].value_counts().head(5)
            results[manufacturer] = handsets
        return results

    @staticmethod
    def aggregate_user_behavior(df):
        """
        Aggregate user behavior metrics: number of xDR sessions, total session duration, 
        total DL, total UL, and total data volume.
        """
        required_columns = ['session_id', 'session_duration', 'total_download', 'total_upload']
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' is missing from the DataFrame.")
        user_behavior = df.groupby('session_id').agg({
            'session_duration': 'sum',
            'total_download': 'sum',
            'total_upload': 'sum'
        }).reset_index()
        user_behavior['total_data_volume'] = user_behavior['total_download'] + user_behavior['total_upload']
        return user_behavior

    @staticmethod
    def compute_dispersion_metrics(df):
        """
        Compute basic metrics for each column in the DataFrame.
        """
        if df.empty:
            raise ValueError("The DataFrame is empty.")
        metrics = df.describe().transpose()
        return metrics[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]

    @staticmethod
    def plot_histogram(df, column, title):
        """
        Plot histogram for a column.
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' is missing from the DataFrame.")
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True, bins=30)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    @staticmethod
    def analyze_application_usage(df, application_column, data_column):
        """
        Explore the relationship between application usage and total data volume.
        """
        if application_column not in df.columns or data_column not in df.columns:
            raise KeyError(f"Columns '{application_column}' or '{data_column}' are missing from the DataFrame.")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[application_column], y=df[data_column])
        plt.title(f'Relationship between {application_column} and {data_column}')
        plt.xlabel(application_column)
        plt.ylabel(data_column)
        plt.show()

    @staticmethod
    def compute_correlation_matrix(df, columns):
        """
        Compute the correlation matrix for specific columns.
        """
        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' is missing from the DataFrame.")
        correlation_matrix = df[columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()
        return correlation_matrix
