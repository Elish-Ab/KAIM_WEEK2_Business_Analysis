import sys
import os

# Add the 'src' directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Now import the module
from src.database import load_data
from src.data_preprocessing import preprocess_data
from src.data_aggregation import aggregate_user_behavior, aggregate_user_engagement
from src.feature_engineering import extract_features
from src.visualization import create_visualizations

def main():
    # Step 1: Load the data from the database
    query = "SELECT * FROM telecom_data"  # Replace with your actual query
    raw_data = load_data(query)
    print("Data loaded successfully")

    # Step 2: Preprocess the data
    preprocessed_data = preprocess_data(raw_data)

    # Step 3: Aggregate data (e.g., per user/application)
    aggregated_data = aggregate_user_behavior(preprocessed_data)

    # Step 4: Extract features (feature engineering step)
    features = extract_features(aggregated_data)

    # Step 5: Visualize the data
    create_visualizations(features)

    print("Analysis Complete!")

if __name__ == "__main__":
    main()
