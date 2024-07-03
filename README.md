# Medical-_-Model-
This project uses Random Forest Classifier models to predict the likelihood of migraines, diabetes, and lung cancer based on patient responses to specific questions, providing tailored advice based on the predictions.
## Disease Prediction Project

### Project Title
Disease Prediction Using Random Forest Classifier

### Description
This project aims to predict the likelihood of various diseases, including migraines, diabetes, and cancer, based on patient responses to a series of questions. The project utilizes a Random Forest Classifier for predictions and provides tailored advice based on the predicted outcomes.

### Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Features](#features)
- [License](#license)

### Project Overview
The project contains three main prediction models:
1. **Migraine Prediction**: Predicts the type of migraine based on patient responses.
2. **Diabetes Prediction**: Predicts the risk of diabetes based on patient responses.
3. **Cancer Prediction**: Predicts the likelihood of lung cancer based on patient responses.

### Installation
To run this project, you will need Python installed on your machine along with the following libraries:
- pandas
- numpy
- scikit-learn

You can install the required libraries using the following command:
```bash
pip install pandas numpy scikit-learn
```

### Usage
1. Clone the repository to your local machine.
2. Ensure you have the required datasets downloaded and accessible.
3. Run the main script to start the prediction process.

```python
python main.py
```

### Project Structure
- `main.py`: The main script to run the prediction models.
- `disease_prediction.py`: Contains the `Diseases` class with methods for transforming data, training models, and predicting diseases.
- `data/`: Directory containing the datasets for migraines, diabetes, and cancer.

### Dataset
The datasets used for this project are sourced from Kaggle and include patient information related to migraines, diabetes, and lung cancer.

- **Migraine Dataset**: [Link to dataset](https://storage.googleapis.com/kagglesdsdata/datasets/2214394/3701345/data.csv)
- **Diabetes Dataset**: [Link to dataset](https://storage.googleapis.com/kagglesdsdata/datasets/2527538/4289678/diabetes.csv)
- **Cancer Dataset**: [Link to dataset](https://storage.googleapis.com/kagglesdsdata/datasets/1623385/2668247/survey%20lung%20cancer.csv)

### Features
1. **Data Transformation**: Encodes categorical data and prepares features and labels.
2. **Model Training**: Trains a Random Forest Classifier for each disease prediction.
3. **User Interaction**: Asks the user a series of questions to gather input data for predictions.
4. **Prediction and Advice**: Predicts the likelihood of a disease and provides tailored advice based on the prediction.
