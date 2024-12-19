import os
from sqlalchemy import create_engine

# Set defaults if environment variables are not found
username = os.getenv('DB_USERNAME', 'postgres')
password = os.getenv('DB_PASSWORD', '12345678')
host = os.getenv('DB_HOST', 'localhost')
port = os.getenv('DB_PORT', '5432')
database = os.getenv('DB_NAME', 'telecom_data')

connection_url = f'postgresql://{username}:{password}@{host}:{port}/{database}'
engine = create_engine(connection_url)

with engine.connect() as conn:
    print("Connection established")

