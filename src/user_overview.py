import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Analyzer:
    # Function to identify top 10 handsets
    def get_top_handsets(self, df, n=10):
        """
        Identify the top N handsets used by customers.
        """
        top_handsets = df['handset_manufacturer'].value_counts().head(n).reset_index()
        top_handsets.columns = ['handset_manufacturer', 'count']
        return top_handsets
    
    @staticmethod
    def plot_top_handsets(df, n=10):
        top_handsets = Analyzer.get_top_handsets(df, n)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_handsets['count'], y=top_handsets['handset_manufacturer'], palette="viridis")
        plt.title(f"Top {n} Handsets")
        plt.xlabel("Count")
        plt.ylabel("Handset Type")
        plt.tight_layout()
        plt.show()

    # Function to identify top 3 manufacturers
    def get_top_manufacturers(self, df, n=3):
        """
        Identify the top N handset manufacturers.
        """
        top_manufacturers = df['manufacturer'].value_counts().head(n).reset_index()
        top_manufacturers.columns = ['manufacturer', 'count']
        return top_manufacturers
    
    @staticmethod
    def plot_top_manufacturers(df, n=3):
        if 'manufacturer' not in df.columns:
            raise KeyError("Column 'manufacturer' is missing from the DataFrame.")
        top_manufacturers = df['manufacturer'].value_counts().head(n)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_manufacturers.values, y=top_manufacturers.index, palette="cubehelix")
        plt.title(f"Top {n} Manufacturers")
        plt.xlabel("Count")
        plt.ylabel("Manufacturer")
        plt.tight_layout()
        plt.show()
    
     # Function to get top 5 handsets per top N manufacturers
    def get_top_handsets_per_manufacturer(self, df, top_manufacturers):
        """
        Identify the top 5 handsets for each of the top manufacturers.
        """
        results = {}
        if isinstance(top_manufacturers, pd.Series):
            top_manufacturers = top_manufacturers.reset_index()  # Ensure it's a DataFrame for indexing

        for manufacturer in top_manufacturers['manufacturer']:
            top_handsets = df[df['handset_manufacturer'] == manufacturer]['handset_manufacturer'].value_counts().head(5)
            results[manufacturer] = top_handsets.reset_index()
            results[manufacturer].columns = ['handset_manufacturer', 'count']
        return results


    
    @staticmethod
    def plot_top_handsets_per_manufacturer(df, top_manufacturers):
        if 'manufacturer' not in df.columns or 'handset_manufacturer' not in df.columns:
            raise KeyError("Columns 'manufacturer' or 'handset_manufacturer' are missing from the DataFrame.")
        # Ensure top_manufacturers is a DataFrame
        if isinstance(top_manufacturers, pd.Series):
            top_manufacturers = top_manufacturers.reset_index()

        for manufacturer in top_manufacturers['manufacturer']:
            handsets = df[df['handset_manufacturer'] == manufacturer]['handset_manufacturer'].value_counts().head(5)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=handsets.values, y=handsets.index, palette="coolwarm")
            plt.title(f"Top 5 Handsets for {manufacturer}")
            plt.xlabel("Count")
            plt.ylabel("Handset Type")
            plt.tight_layout()
            plt.show()

    
    # Function to aggregate user behavior metrics
    def aggregate_user_behavior(self, df):
        """
        Aggregate user behavior metrics: number of xDR sessions, total session duration, 
        total DL, total UL, and total data volume.
        """
        user_behavior = df.groupby('user_id').agg(
            num_sessions=('session_id', 'count'),
            total_duration=('session_duration', 'sum'),
            total_download=('download', 'sum'),
            total_upload=('upload', 'sum')
        ).reset_index()
        user_behavior['total_data_volume'] = user_behavior['total_download'] + user_behavior['total_upload']
        return user_behavior
    
    # Function for decile segmentation
    def segment_users_by_decile(self, df):
        """
        Segment users into decile classes based on total session duration.
        """
        df['decile'] = pd.qcut(df['total_duration'], 10, labels=False)
        decile_summary = df.groupby('decile').agg(
            total_data=('total_data_volume', 'sum')
        ).reset_index()
        return decile_summary

    # Non-graphical univariate analysis
    def compute_dispersion_metrics(self, df):
        """
        Compute basic metrics for each column in the DataFrame.
        """
        return df.describe()

    # Graphical univariate analysis
    @staticmethod
    def plot_histogram(df, column, title):
        """
        Plot histogram for a column.
        """
        plt.hist(df[column], bins=30, alpha=0.7)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    # Function to analyze relationship between applications and total data
    def analyze_application_usage(self, df, application_column, data_column):
        """
        Explore the relationship between application usage and total data volume.
        """
        sns.scatterplot(data=df, x=application_column, y=data_column)
        plt.title(f'Relationship between {application_column} and {data_column}')
        plt.xlabel(application_column)
        plt.ylabel(data_column)
        plt.show()

    # Function to compute correlation matrix
    def compute_correlation_matrix(self, df, columns):
        """
        Compute the correlation matrix for specific columns.
        """
        corr_matrix = df[columns].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
        return corr_matrix
