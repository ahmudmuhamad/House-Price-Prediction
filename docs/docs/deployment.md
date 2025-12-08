# AWS Deployment Guide

This project uses a Serverless Container architecture.

## Infrastructure Components
* **ECR (Elastic Container Registry):** Stores private Docker images for API and UI.
* **ECS (Elastic Container Service):** Orchestrates the containers using Fargate.
* **S3 (Simple Storage Service):** Stores the Model (`.pkl`) and Data Artifacts.

## CI/CD Workflow
We use GitHub Actions to automate deployment.

1.  **Test:** Runs `pytest` on push.
2.  **Build:** Builds Docker images for API and UI.
3.  **Push:** Uploads images to ECR.
4.  **Deploy:** Forces a new deployment on ECS Services.

!!! warning "Cost Management"
    The `desired_count` for services is set to **0** by default to prevent AWS billing charges. Set to **1** via AWS Console to launch the app.
