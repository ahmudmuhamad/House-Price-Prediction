# System Architecture

## Overview
The system follows a "Training-Serving Skew" prevention pattern by saving transformation artifacts (Imputers, Encoders) alongside the model.

```mermaid
graph LR
    Raw[Raw CSV] --> Clean[Clean.py]
    Clean --> Transform[Transform.py]
    Transform -->|Save| Art[Artifacts Store <br> (S3)]
    Transform --> Train[Train.py]
    Train -->|Save| Model[Model.pkl]
    Model --> Art
    Art --> API[FastAPI Container]
    Art --> UI[Streamlit Container]
```

## Storage Strategy
We use **Amazon S3** as the centralized artifact store.
* **`data/processed/`**: Holds the training data used for context enrichment.
* **`models/artifacts/`**: Holds `imputer.json` and `freq_map.pkl` to ensure the API processes data exactly like the training script.
