import os
import requests
from pymilvus import Collection, connections

# Configurations
IMAGE_FOLDER = "./images"
BENTOML_URL = "http://127.0.0.1:3000/vectorize"
COLLECTION_NAME = "image_vectors"

# Milvus connection
connections.connect("default", host="127.0.0.1", port="19530")
collection = Collection(COLLECTION_NAME)

# Check if the collection exists
if not collection.is_empty:
    print(f"Collection {COLLECTION_NAME} already exists and contains data.")

# Process and insert images
def process_and_insert_images():
    image_names = []
    vectors = []

    for image_file in os.listdir(IMAGE_FOLDER):
        if image_file.lower().endswith(('.jpg', '.jpeg')):
            image_path = os.path.join(IMAGE_FOLDER, image_file)
            print(f"Processing {image_path}...")

            # Call BentoML service to get vector
            with open(image_path, "rb") as image:
                response = requests.post(
                    BENTOML_URL,
                    headers={"Content-Type": "image/jpeg"},
                    data=image
                )

            if response.status_code == 200 and "vector" in response.json():
                image_names.append(image_file)
                vectors.append(response.json()["vector"])
            else:
                print(f"Failed to get vector for {image_file}. Response: {response.text}")

    # Insert into Milvus
    if image_names and vectors:
        collection.insert([image_names, vectors])
        print(f"Inserted {len(image_names)} images into Milvus.")
    else:
        print("No data to insert.")


# Run the process
process_and_insert_images()

collection.create_index(
    field_name="vector",
    index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
)
collection.load()


# Flush and load collection
collection.flush()
collection.load()
print(f"Collection {COLLECTION_NAME} is ready for querying!")
