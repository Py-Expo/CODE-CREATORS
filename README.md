##Readme file 


# Heart Transplant Prediction

This Python script predicts the success of a heart transplant based on user input data using a pre-trained neural network model.

## Prerequisites

- Python 3.x
- TensorFlow
- Pandas
- Scikit-learn

## Installation

1. Clone the repository:

git clone https://github.com/your_username/heart-transplant-prediction.git


2. Install the required dependencies:

3.
## Dataset

The dataset used for training the model is named heart_transplant_data.csv. It contains the following columns:

- HLA: Human leukocyte antigen
- Blood Type: Blood type of the patient
- Gender: Gender of the patient
- BMI: Body Mass Index
- Age: Age of the patient
- Transplant Success: Target variable indicating the success of heart transplant (1 for success, 0 for failure)

You can find the dataset in the project directory.

## Usage

1. Run the Python script:


2. Enter the required input data when prompted:

- HLA
- Blood Type
- Gender
- BMI
- Age

3. The script will output the prediction result.

## Model

The neural network model architecture is as follows:

1. Input layer with 64 neurons and ReLU activation function.
2. Hidden layer with 32 neurons and ReLU activation function.
3. Output layer with 1 neuron and sigmoid activation function.

The model is compiled using the Adam optimizer and binary cross-entropy loss function.

## Additional Notes

- The saved model weights are loaded from 'heart_transplant_model.h5' file.
- Categorical variables are encoded using LabelEncoder.
- Numerical features are normalized using StandardScaler.
