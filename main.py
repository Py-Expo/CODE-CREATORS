from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights('heart_transplant_model.h5')

# Load label encoders and scaler
label_encoders = {}
scaler = StandardScaler()
categorical_cols = ['HLA', 'Blood Type', 'Gender']
numerical_cols = ['BMI', 'Age']
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()

@app.route('/')
def index():
    return render_template('index6.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Create a DataFrame with user input
    input_data = pd.DataFrame({
        'HLA': [data['hla']],
        'Blood Type': [data['blood_type']],
        'Gender': [data['gender']],
        'BMI': [float(data['bmi'])],
        'Age': [float(data['age'])]
    })

    # Convert categorical variables into numerical representations
    for col in categorical_cols:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Normalize numerical features
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    # Make predictions
    prediction = model.predict(input_data)
    if prediction > 0.5:
        result = "Transplant success"
    else:
        result = "Not transplant success"
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
