# CLIP Image Vectorizer with BentoML

This project provides an API for vectorizing images using OpenAI's CLIP model, packaged and served with BentoML. It allows you to send images to the API and receive a vector representation that encodes meaningful features of the image.

## Features
- Easy-to-use REST API for image vectorization.
- Powered by the OpenAI CLIP model.
- Serves vectorization directly from BentoML.
- Ready for local testing.

---

## Install

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   python3 -m pip install transformers Pillow torch bentoml
   ```

---

## Run a Local Service

1. Start the BentoML service:
   ```bash
   bentoml serve service:svc
   ```
2. Verify the service is running:
   - Health Check:
     ```bash
     curl http://127.0.0.1:3000/livez
     ```
   - Metrics:
     ```bash
     curl http://127.0.0.1:3000/metrics
     ```

---

## Vectorize an Image

Send an image to the API and receive a vector representation:

```bash
curl -X POST -H "Content-Type: image/jpeg" --data-binary @image.jpg http://127.0.0.1:3000/vectorize
```

The response will be a JSON object containing the image vector.
