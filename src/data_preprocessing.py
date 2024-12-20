import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Process:
    # Function to handle missing values
    @staticmethod
    def handle_missing_values(df):
        # Handle numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        df[numeric_df.columns] = numeric_df.fillna(numeric_df.mean())

        # Handle categorical columns
        categorical_df = df.select_dtypes(include=['object'])
        for col in categorical_df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

        return df


    # Function to handle outliers using IQR
    @staticmethod
    def remove_outliers(df, column):
        """
        Remove outliers based on the Interquartile Range (IQR) method for a specific column.
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        return df

    # Function to perform PCA
    @staticmethod
    def perform_pca(df, n_components=2):
        """
        Perform Principal Component Analysis (PCA) on the dataset.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        
        explained_variance = pca.explained_variance_ratio_
        return principal_components, explained_variance
