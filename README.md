# **CLIP Image Vectorizer with BentoML and Milvus**

This project offers an API for vectorizing images using OpenAI's CLIP model via BentoML. By sending images to the API, you receive a vector representation that encodes meaningful features of the image.

In addition, this project integrates Milvus for efficient vector storage and search, alongside MinIO and etcd to provide a more robust, complete system.

---

## **Features**
- REST API for image and text vectorization.
- Powered by OpenAI CLIP, served through BentoML.
- Docker Compose integration for a full-stack setup with Milvus, MinIO, and etcd.
- Python scripts for:
  - Importing image vectors into Milvus.
  - Performing similarity searches on image vectors.

---

## **Getting Started**

### **1. Set Up the Environment**

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install transformers Pillow torch bentoml
   ```

---

## **Running BentoML Locally**

### **Start BentoML Service**
1. Launch the BentoML service:
   ```bash
   bentoml serve service:vectorization
   ```

2. Verify the service:
   - Health Check:
     ```bash
     curl -v http://127.0.0.1:3000/livez
     ```
   - Metrics:
     ```bash
     curl http://127.0.0.1:3000/metrics
     ```

### **Vectorize Inputs**

1. **Vectorize a string**:
   ```bash
   curl -X POST -H "Content-Type: text/plain" -d "dog" http://127.0.0.1:3000/vectorize_text
   ```

2. **Vectorize an image**:
   ```bash
   curl -X POST -H "Content-Type: image/jpeg" --data-binary @image.jpg http://127.0.0.1:3000/vectorize_image
   ```
   The response will be a JSON object containing the image vector.

---

## **Building and Running the Container**

1. **Build and tag the BentoML container**:
   ```bash
   bentoml build
   bentoml containerize clip_image_vectorizer:latest -t bentoml:latest
   ```

2. **Run the container**:
   ```bash
   docker run --rm -p 3000:3000 bentoml:latest
   ```

---

## **Full Stack: BentoML + Milvus**

### **Run the Stack**
1. Use Docker Compose to start all services:
   ```bash
   docker compose up -d
   ```

2. Verify service statuses:
   - **etcd**:
     ```bash
     curl -X GET "http://127.0.0.1:2379/health"
     ```
   - **Milvus**:
     ```bash
     curl -X GET "http://127.0.0.1:9091/api/v1/health"
     ```
   - **List Milvus collections**:
     ```bash
     curl -X GET "http://127.0.0.1:9091/api/v1/collections"
     ```

---

## **Image Vector Management with Milvus**

### **1. Import Vectors**

1. Create a folder `/images` and add image files (`.jpg`, `.jpeg`).
2. Run the Python import script:
   ```bash
   python3 milvus_import.py
   ```

   This script:
   - Creates a Milvus collection.
   - Imports vectors of the images in `/images` into Milvus.

---

### **2. Search for Similar Images**

1. Update the search term in `milvus_search.py`.
2. Run the search script:
   ```bash
   python3 milvus_search.py
   ```

   Example output:
   ```
   Using L2 (Euclidean distance) for search...
   Search results:
   ID: image1.jpg, Score: 171.71
   ID: image2.jpeg, Score: 174.32
   ...
   ```

   Lower scores indicate closer matches.

---

## **Supported Image Formats**
This service supports the following image formats via Pillow:

| **Format**      | **File Extensions**         | **Description**                                                              |
|------------------|-----------------------------|------------------------------------------------------------------------------|
| **JPEG**        | `.jpg`, `.jpeg`             | Common format, widely used for photographs.                                  |
| **PNG**         | `.png`                      | Lossless compression, supports transparency.                                 |
| **BMP**         | `.bmp`                      | Bitmap image format, uncompressed.                                          |
| **GIF**         | `.gif`                      | Supports animation; processes the first frame.                              |
| **TIFF**        | `.tiff`, `.tif`             | Flexible format with compression options.                                   |
| **WEBP**        | `.webp`                     | Modern image format for web usage.                                          |
| **HDR**         | `.hdr`                      | High Dynamic Range images.                                                  |
| **TGA**         | `.tga`                      | Common in video games and graphics.                                         |

---