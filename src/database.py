import pandas as pd
import psycopg2

def load_data(query):
    try:
        # Establishing connection
        conn = psycopg2.connect("dbname=telecom_data user=postgres password=12345678 host=localhost")
        # Running query
        df = pd.read_sql(query, conn)
        conn.close()

        # Debugging the loaded data
        if df is None:
            print("No data returned from the query.")
        else:
            print(f"Data loaded with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
