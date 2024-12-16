from pymilvus import connections, Collection
import requests
import json

# Constants
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "image_vectors"
BENTOML_TEXT_VECTORIZE_URL = "http://127.0.0.1:3000/vectorize_text"
SEARCH_TEXT = "a bed"

# Connect to Milvus
print("Connecting to Milvus...")
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# Load Milvus Collection
print(f"Loading collection '{COLLECTION_NAME}'...")
collection = Collection(COLLECTION_NAME)
collection.load()

# Get the vector for the search text using BentoML
print(f"Vectorizing the search text '{SEARCH_TEXT}'...")
response = requests.post(
    BENTOML_TEXT_VECTORIZE_URL,
    headers={"Content-Type": "application/json"},
    data=json.dumps({"text": SEARCH_TEXT})
)

if response.status_code != 200:
    print(f"Failed to vectorize text. Response: {response.text}")
    exit()

query_vector = response.json().get("vector")
if not query_vector:
    print("Vectorization failed: No vector returned.")
    exit()

print(f"Query vector: {query_vector}")

# Perform search in Milvus
print("Searching the Milvus collection...")
search_results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "L2", "params": {"nprobe": 2}},  # Ensure 'L2' matches the collection's metric_type
    limit=100,
    expr=""
)

# Display results
print("Search results:")
for hits in search_results:
    for hit in hits:
        print(f"ID: {hit.id}, Score: {hit.score}, Data: {hit.entity}")

print("Search completed.")
