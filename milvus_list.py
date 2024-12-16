from pymilvus import Collection, utility, connections

COLLECTION_NAME = "image_vectors"

# Milvus connection
connections.connect("default", host="127.0.0.1", port="19530")
collection = Collection(COLLECTION_NAME)

# Check if collection exists
if not utility.has_collection(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' does not exist!")
else:
    # Load collection into memory
    collection.load()

    # List all inserted data (max limit for query is 10,000)
    print("Querying all data in the collection...")
    results = collection.query(expr="", output_fields=["image_name", "vector"], limit=10)
    for item in results:
        print(item)
