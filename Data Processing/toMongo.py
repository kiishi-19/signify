import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

connection_key = "MONGO_URL"

csv_file_path = '../data/'

database_name = 'Signyfy'
collection_name = 'hello'

client = MongoClient(connection_key)

try:
    client.admin.command('ping')
    print("Connected to MongoDB Atlas")
except ConnectionFailure:
    print("Connection failed")
    exit()

data = pd.read_csv(csv_file_path)

data_dict = data.to_dict("records")

db = client[database_name]

collection = db[collection_name]

collection.insert_many(data_dict)
print(f"Inserted {len(data_dict)} records into {database_name}.{collection_name}")

client.close()
