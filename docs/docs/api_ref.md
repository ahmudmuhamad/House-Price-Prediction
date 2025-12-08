# API Reference

This section documents the source code modules for the House Price Prediction system.

## Feature Pipeline
These modules handle data loading, cleaning, and feature engineering.

### Data Loading
::: src.feature_pipeline.load

### Data Cleaning
::: src.feature_pipeline.clean

### Transformation
::: src.feature_pipeline.transform

## Training Pipeline
These modules manage model training, hyperparameter tuning, and evaluation.

### Tuning
::: src.training_pipeline.tune

### Training
::: src.training_pipeline.train

### Evaluation
::: src.training_pipeline.eval

## Inference Pipeline
These modules handle the prediction logic for new data.

### Inference
::: src.inference_pipeline.inference

## API
These modules define the FastAPI application and data schemas.

### Main Application
::: src.api.main

### Schemas
::: src.api.schemas

## Utilities
Helper scripts and functions.

### S3 Upload
::: src.utils.upload_to_s3
