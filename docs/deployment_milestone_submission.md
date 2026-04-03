# Deployment Milestone Submission Template

Fill in the placeholders below, export to PDF, and submit that PDF for the milestone.

## 1. GitHub Repository

- Repository URL: `PASTE_GITHUB_REPO_URL_HERE`
- Endpoint file: `PASTE_DIRECT_LINK_TO_src/clothing_mlops/service.py_HERE`

## 2. Docker Image Proof

- Published image name: `YOUR_DOCKERHUB_USERNAME/clothing-value-api:latest`
- Registry URL: `PASTE_DOCKER_IMAGE_URL_HERE`
- Screenshot: insert a screenshot of the published image page showing the exact image name above

## 3. Run Instructions Summary

Local API:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -e .
python scripts/build_sample_dataset.py
python scripts/train_model.py
export MLFLOW_SERVING_MODEL_URI="models:/clothing-value-model/latest"
uvicorn clothing_mlops.service:app --reload
```

Docker:

```bash
docker build -t clothing-value-api .
docker run -p 8000:8000 \
  -v "$(pwd)/mlruns:/app/mlruns" \
  -e MLFLOW_TRACKING_URI="file:/app/mlruns" \
  -e MLFLOW_SERVING_MODEL_URI="models:/clothing-value-model/latest" \
  clothing-value-api
```

## 4. Example Predict Request / Response

Request:

```json
{
  "brand": "Patagonia",
  "category": "Jacket",
  "size": "M",
  "condition": "used_very_good",
  "color": "Blue",
  "material": "Polyester",
  "listing_price": 129.0,
  "shipping_price": 12.5
}
```

Response:

```json
{
  "predicted_sale_price": 126.13,
  "model_status": "ok"
}
```
