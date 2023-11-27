## YouTube Virality Prediction Project
### Introduction
Welcome to the YouTube Virality Prediction project! This comprehensive machine learning end-to-end solution is designed to predict the virality of YouTube videos based on various features such as views, likes, dislikes, title, description, tags, publish date, trending date, and comments. Leveraging state-of-the-art technologies in the field, this project utilizes ZenML as the MLOps framework, MLflow for experiment tracking and model registration, Evidently for data validation, and Streamlit for a user-friendly app interface. The entire workflow is seamlessly integrated with zenml for efficient management of ML life cycle.

#### YouTube Statistics
Before diving into the project details, let's take a moment to appreciate the impact and scale of YouTube:

1. YouTube has over 2 billion logged-in monthly users (YouTube Press).
2. More than 500 hours of video content are uploaded every minute (YouTube Press).
3. The platform is available in 100+ countries and supports 80 languages (YouTube Press).
4. It reaches more 18-49 year-olds than any single cable network in the United States (YouTube Press).

### Problem Statement
The explosive growth of content on YouTube makes it challenging for creators and marketers to predict the virality of their videos accurately. This project addresses this problem by developing a robust machine learning model that takes various video attributes into account to predict the likelihood of a video going viral. By providing insights into the key factors influencing virality, this project empowers content creators to optimize their content strategy for increased visibility and engagement.

### Getting Started
To get started with the YouTube Virality Prediction project, follow these steps:

#### 1. Clone the Repository:

```bash
git clone https://github.com/your-username/youtube-virality-prediction.git
cd youtube-virality-prediction
```

#### 2. Install the dependencies:
```bash
pip install -r requirements.txt
```

#### 3. Set Up ZenML:
The project uses ZenML for managing the MLOps workflow. Ensure you have ZenML installed and configured.
```bash
pip install "zenml[server]"
```

### Setup Zenml stack
```bash
#installing evidently for data validation
zenml integration install evidently -y
#Register the Evidently data validator
zenml data-validator register evidently_data_validator --flavor=evidently

#nstalling evidently for model-deployement
zenml integration install mlflow -y
#To register the MLflow model deployer with ZenML
zenml model-deployer register mlflow_deployer --flavor=mlflow


# Register the MLflow experiment tracker
zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow

# Register the mlflow model_registry
zenml model-registry register mlflow_model_registry --flavor=mlflow

# Register and set a stack with the new model registry as the active stack
zenml stack register custom_stack -r mlflow_model_registry -e mlflow_experiment_tracker -dv evidently_data_validator -a default -o default -d mlflow_deployer --set
```

This stack will help to control the end to end Machine Learning lifecycle of this project.

In this project we are using Azure for fetching the data from cloud.
We upload the data to azure using data_uploading.py

``` bash
python data_uploading.py
```

### Pipelines
In this project we use 2 pipelines

#### 1. Training pipeline:
In this pipeline first we we fetch data from cloud, then preprocess the data using cleaning_data step,
then we split the data using the step sklearn_split_data, train the model with 3 machine learning the models
1. Random Forest,
2. Ada Boost Classifier
3. Gradient Boosting

then evaluating the accuracy score and selecting the best model. This model will be registered in MLflow model registry. Finally we will monitor the model using drift reports.

#### 2. Deployment pipeline
We ingest the data, preprocess the data, splitting the data, deploying the model, and make predictions using the deployed model

To run these pipelines
``` bash
python run_pipeline.py

```

You can predict the virality of a video by input the data using the streamlit web app.

run
```bash
streamlit run streamlit.py
```