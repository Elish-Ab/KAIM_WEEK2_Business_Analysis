import pandas as pd
from src.database import load_data
from src.user_overview import Analyzer
from src.data_preprocessing import Process

# Load data from the PostgreSQL database
query = """
    SELECT 
        "Bearer Id", "Start", "End", "Dur. (ms)", "IMSI", "MSISDN/Number", "IMEI", 
        "Last Location Name", "Avg RTT DL (ms)", "Avg RTT UL (ms)", "Avg Bearer TP DL (kbps)",
        "Avg Bearer TP UL (kbps)", "TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)",
        "DL TP < 50 Kbps (%)", "50 Kbps < DL TP < 250 Kbps (%)", "250 Kbps < DL TP < 1 Mbps (%)",
        "DL TP > 1 Mbps (%)", "UL TP < 10 Kbps (%)", "10 Kbps < UL TP < 50 Kbps (%)", 
        "50 Kbps < UL TP < 300 Kbps (%)", "UL TP > 300 Kbps (%)", "HTTP DL (Bytes)", 
        "HTTP UL (Bytes)", "Activity Duration DL (ms)", "Activity Duration UL (ms)", 
        "Handset Manufacturer", "Handset Type", "Social Media DL (Bytes)", "Social Media UL (Bytes)",
        "Google DL (Bytes)", "Google UL (Bytes)", "Email DL (Bytes)", "Email UL (Bytes)", 
        "Youtube DL (Bytes)", "Youtube UL (Bytes)", "Netflix DL (Bytes)", "Netflix UL (Bytes)",
        "Gaming DL (Bytes)", "Gaming UL (Bytes)", "Other DL (Bytes)", "Other UL (Bytes)", 
        "Total DL (Bytes)", "Total UL (Bytes)"
    FROM "xdr_data"
"""

try:
    # Ensure no parameters are passed explicitly
    df = load_data(query)
    print(f"Data loaded with shape: {df.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Standardize column names
try:
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    print("Standardized columns:", df.columns)
except Exception as e:
    print(f"Error standardizing columns: {e}")
    exit()

# Initialize the analyzer
analyzer = Analyzer()

# Data Preprocessing: Handle missing values and outliers
try:
    process = Process()

    # Handle missing values
    df = process.handle_missing_values(df)
    print("Missing values handled.")

    # Handle outliers for relevant columns
    df = process.remove_outliers(df, 'dur._(ms)')
    print("Outliers handled.")
except KeyError as e:
    print(f"Error during preprocessing: {e}")
    exit()
except Exception as e:
    print(f"Unexpected error during preprocessing: {e}")
    exit()

# Exploratory Data Analysis (EDA): User Overview Analysis
try:
    # 1. Get Top 10 Handsets
    top_handsets = analyzer.get_top_handsets(df)
    analyzer.plot_top_handsets(df)
    print("Top 10 Handsets:\n", top_handsets)

    # 2. Get Top 3 Manufacturers
    top_manufacturers = analyzer.get_top_manufacturers(df)
    analyzer.plot_top_handsets_per_manufacturer(df)
    print("\nTop 3 Manufacturers:\n", top_manufacturers)

    # 3. Get Top 5 Handsets Per Manufacturer
    analyzer.plot_top_handsets_per_manufacturer(df,top_manufacturers)


    # 4. Aggregate User Behavior
    user_behavior = analyzer.aggregate_user_behavior(df)
    print("\nAggregated User Behavior:\n", user_behavior.head())

    # 5. Compute Dispersion Metrics
    dispersion_metrics = analyzer.compute_dispersion_metrics(df)
    print("\nDispersion Metrics:\n", dispersion_metrics)

    # 6. Plot Histogram of Session Duration
    analyzer.plot_histogram(df, 'dur._(ms)', 'Session Duration Distribution')

    # 7. Analyze Application Usage
    analyzer.analyze_application_usage(df, 'social_media_dl_(bytes)', 'total_dl_(bytes)')
    analyzer.plot_application_usage(df,'social_media_dl_(bytes)', 'total_dl_(bytes)')
    # 8. Compute Correlation Matrix for Application Usage
    correlation_matrix = analyzer.compute_correlation_matrix(df, 
                                                           columns=['social_media_dl_(bytes)', 'google_dl_(bytes)', 
                                                                    'email_dl_(bytes)', 'youtube_dl_(bytes)', 
                                                                    'netflix_dl_(bytes)', 'gaming_dl_(bytes)'])
    print("\nCorrelation Matrix:\n", correlation_matrix)
    analyzer.plot_correlation_matrix(df,columns=['social_media_dl_(bytes)', 'google_dl_(bytes)', 
                                                                    'email_dl_(bytes)', 'youtube_dl_(bytes)', 
                                                                    'netflix_dl_(bytes)', 'gaming_dl_(bytes)'])

except KeyError as e:
    print(f"Missing column for analysis: {e}")
except Exception as e:
    print(f"Unexpected error during analysis: {e}")
    exit()

# 9. Perform PCA
try:
    principal_components, explained_variance = process.perform_pca(df[['dur._(ms)', 'total_dl_(bytes)', 'total_ul_(bytes)']], n_components=2)
    print("\nPCA Results - Principal Components:\n", principal_components)
    print("\nExplained Variance Ratio:\n", explained_variance)
except KeyError as e:
    print(f"Missing column for PCA: {e}")
except Exception as e:
    print(f"Unexpected error during PCA: {e}")
    exit()
