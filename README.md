# Network Security ML Project: Phishing Detection

This project implements an end-to-end machine learning pipeline for phishing URL detection using MLOps best practices. It covers data ingestion, validation, transformation, model training, and deployment with a web interface.

## Project Structure

- `network_security/components/`
  - `data_ingestion.py`: Loads data from MongoDB and exports to feature store.
  - `data_validation.py`: Validates schema, checks for numerical columns, and detects data drift.
  - `data_transformation.py`: Cleans and transforms data, applies imputation, and prepares features for modeling.
  - `data_trainer_model.py`: Trains multiple ML models and selects the best one.
- `templates/table.html`: Renders prediction results in a web table.
- `main.py`: Orchestrates the pipeline steps.
- `.env`: Stores environment variables (e.g., MongoDB connection string).

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repo_url>
   cd mlops_project
   ```

2. **Install dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   - Create a `.env` file in the root directory:
     ```
     MONGO_DB_URL=<your_mongodb_connection_string>
     ```

4. **Run the pipeline**
   ```bash
   python main.py
   ```

5. **Start the web server (if applicable)**
   ```bash
   flask run
   ```
   - Visit `http://localhost:5000` to access the prediction interface.

## EC2 & GitHub Actions Setup
- In order to run the github workflow, commit the changes and go to the github action and run the workflow.


- Update and install Docker in EC2 instance:
  ```bash
  sudo apt-get update -y && sudo apt-get upgrade -y
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  sudo usermod -aG docker ubuntu
  newgrp docker
  ```

- Set up GitHub Actions runner:
  ```bash
  mkdir actions-runner && cd actions-runner
  # Download latest runner package
  curl -o actions-runner.tar.gz -L https://github.com/actions/runner/releases/download/<version>/actions-runner-linux-x64-<version>.tar.gz
  # Validate the hash
  echo "<hash_id> actions-runner-linux-x64-<version>.tar.gz" | shasum -a 256 -c
  tar xzf ./actions-runner-linux-x64-<version>.tar.gz
  # Configure and run
  ./config.sh --url <github_repo> --token <token>
  ./run.sh
  ```

## Notes

- Ensure MongoDB Atlas or local MongoDB is running and accessible.
- Update schema and config files as per your dataset.
- For troubleshooting, check logs and exception messages.


### Run the github workflow

### Run the following commands in EC2 instance:
- sudo apt-get update -y && sudo apt-get upgrade -y
- curl -fssL https://get.docker.com -o get-docker.sh
- sudo get-docker.sh
- sudo usermod -aG docker ubuntu
- newgrp docker

Refer github actions runner in gitub for below command:
- mkdir actions-runner && cd actions-runner
- Download latest runner package: curl -o actions-runner-...-tar.gz -L https://github.com/actions/runner/release/download/{version}/actions-runner-linux-x64-{version}.tar.gz
- Validate the hash: echo "{hash_id} actions-runner-linux-x64-{version}.tar.gz" | shasum -a 256 -c
- tar xzf ./actions-runner-linux-x64-{version}.tar.gz
- Configure: 
    - ./config.sh --url <github_repo> --token <token>  . Note: name of runner is self-hosted here.
    - ./run.sh
