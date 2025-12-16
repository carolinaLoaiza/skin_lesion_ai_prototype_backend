# Skin Lesion AI Prototype Backend

Academic prototype backend API for skin lesion malignancy risk prediction using multiple AI models. Developed as part of a Master's AI dissertation project.

## Overview

This API processes skin lesion images and clinical metadata to predict malignancy risk using an ensemble of three machine learning models:

- **Model A**: Deep learning image classifier (direct probability prediction)
- **Model B**: Feature extractor (18 image features + diameter)
- **Model C**: Tabular classifier using Model B features + clinical metadata

### Prediction Pipeline

```
Input (Image + Metadata)
    ↓
Preprocessing
    ↓
Model A → Probability A
    ↓
Model B → Extracted Features
    ↓
Model C (Features + Metadata) → Probability C
    ↓
Weighted Average (A + C) → Final Probability
```

## Project Structure

```
skin_lesion_ai_prototype_backend/
├── app/
│   ├── __init__.py
│   ├── api/                    # API endpoints
│   │   ├── __init__.py
│   │   └── prediction.py       # Prediction endpoint
│   ├── core/                   # Core configuration
│   │   ├── __init__.py
│   │   ├── config.py           # Settings and environment variables
│   │   └── logger.py           # Logging configuration
│   ├── models/                 # Model loading and inference
│   │   └── __init__.py
│   ├── schemas/                # Pydantic models for validation
│   │   ├── __init__.py
│   │   └── prediction.py       # Request/Response schemas
│   ├── services/               # Business logic and pipeline
│   │   └── __init__.py
│   └── utils/                  # Utility functions
│       └── __init__.py
├── tests/                      # Unit and integration tests
│   └── __init__.py
├── saved_models/               # Saved model files (not in git)
├── logs/                       # Application logs
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore
└── README.md
```

## Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository:
```bash
cd skin_lesion_ai_prototype_backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Create environment file:
```bash
# Copy the example file
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac

# Edit .env with your configuration
```

6. Place model files in `saved_models/` directory:
```
saved_models/
├── model_a.h5      # Deep learning model
├── model_b.h5      # Feature extractor
└── model_c.pkl     # Tabular classifier
```

## Running the Application

### Development Mode

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once the application is running, visit:

- **Interactive API docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative API docs (ReDoc)**: http://localhost:8000/redoc

## API Endpoints

### POST /api/predict

Predict malignancy risk for a skin lesion.

**Request:**
- `image` (file): Skin lesion image (JPEG/PNG)
- `age` (int): Patient age in years (0-120)
- `sex` (string): Patient sex ("male" or "female")
- `location` (string): Lesion location (e.g., "back", "arm", "face")
- `diameter` (float): Lesion diameter in millimeters

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "image=@lesion.jpg" \
  -F "age=45" \
  -F "sex=female" \
  -F "location=back" \
  -F "diameter=6.5"
```

**Example using Python:**
```python
import requests

url = "http://localhost:8000/api/predict"
files = {"image": open("lesion.jpg", "rb")}
data = {
    "age": 45,
    "sex": "female",
    "location": "back",
    "diameter": 6.5
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Response:**
```json
{
  "final_probability": 0.65,
  "model_a_probability": 0.72,
  "model_c_probability": 0.58,
  "extracted_features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
  "metadata": {
    "age": 45,
    "sex": "female",
    "location": "back",
    "diameter": 6.5
  }
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## Configuration

Key configuration options in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_NAME` | Application name | skin_lesion_ai_prototype_backend |
| `DEBUG` | Debug mode | True |
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 8000 |
| `MODEL_A_PATH` | Path to Model A | saved_models/model_a.h5 |
| `MODEL_B_PATH` | Path to Model B | saved_models/model_b.h5 |
| `MODEL_C_PATH` | Path to Model C | saved_models/model_c.pkl |
| `MODEL_A_WEIGHT` | Weight for Model A in final prediction | 0.5 |
| `MODEL_C_WEIGHT` | Weight for Model C in final prediction | 0.5 |
| `IMAGE_SIZE` | Image size for processing | 224 |
| `LOG_LEVEL` | Logging level | INFO |

## Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_prediction.py
```

## Development

### Code Style

- Follow PEP 8 guidelines
- Use snake_case for functions, variables, and file names
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions small and focused

### Adding New Features

1. Create appropriate schemas in `app/schemas/`
2. Implement business logic in `app/services/`
3. Add utility functions in `app/utils/`
4. Create API endpoints in `app/api/`
5. Write tests in `tests/`

## License

Academic prototype - Master's dissertation project

## Contact

For questions or issues, please contact the project maintainer.
