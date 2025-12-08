# Getting Started

Follow these instructions to set up the project environment and run the pipeline.

## Prerequisites
*   Python 3.11+
*   `uv` (Universal Python Package Manager)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ahmudmuhamad/House-Price-Prediction.git
    cd House-Price-Prediction
    ```

2.  **Install dependencies:**
    This project uses `make` and `uv` for easier management.
    ```bash
    make requirements
    ```

## Running the Pipeline

### Data Processing
To process the data:
```bash
make data
```

### Running Tests
To run the test suite:
```bash
make test
```

### Running the API
To run the FastAPI server with hot-reload:
```bash
uv run uvicorn src.api.main:app --reload
```

