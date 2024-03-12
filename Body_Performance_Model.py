import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def predict_body_performance(age, gender, height_cm, weight_kg, body_fat_percentage, diastolic, systolic, grip_force, sit_and_bend_forward_cm, sit_ups_counts, broad_jump_cm):
    # Load the dataset
    df = pd.read_csv(r'C:\Users\sgoff\Downloads\bodyPerformance (1).csv')

    # Encode gender and class columns
    label_encoder = LabelEncoder()
    df['gender_encoded'] = label_encoder.fit_transform(df['gender'])
    df['class_encoded'] = label_encoder.fit_transform(df['class'])
    df.drop(columns=['gender', 'class'], inplace=True)

    # Define features
    Features = df[['age', 'height_cm', 'weight_kg', 'body fat_%', 'diastolic', 'systolic', 'gripForce', 'sit and bend forward_cm', 'sit-ups counts', 'broad jump_cm', 'gender_encoded']]

    # Define target variable
    Target = df['class_encoded']

    # Initialize the model
    model = Sequential([
        Dense(200, input_dim=11, activation='relu'),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(10, activation='relu'),
        Dense(4, activation='softmax')
    ])

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Prepare input data for prediction
    input_data = np.array([[age, height_cm, weight_kg, body_fat_percentage, diastolic, systolic, grip_force, sit_and_bend_forward_cm, sit_ups_counts, broad_jump_cm, 0]])

    # Make prediction
    prediction = model.predict(input_data)
    predicted_class_index = np.argmax(prediction)
    
    # Decode the predicted class
    classes = ['A', 'B', 'C', 'D']
    predicted_class = classes[predicted_class_index]

    return predicted_class

# Example usage:
if _name_ == "_main_":
    age = float(input("Enter age: "))
    gender = input("Enter gender (M/F): ")
    height_cm = float(input("Enter height in cm: "))
    weight_kg = float(input("Enter weight in kg: "))
    body_fat_percentage = float(input("Enter body fat percentage: "))
    diastolic = float(input("Enter diastolic blood pressure: "))
    systolic = float(input("Enter systolic blood pressure: "))
    grip_force = float(input("Enter grip force: "))
    sit_and_bend_forward_cm = float(input("Enter sit and bend forward measurement in cm: "))
    sit_ups_counts = float(input("Enter sit-ups counts: "))
    broad_jump_cm = float(input("Enter broad jump distance in cm: "))

    predicted_class = predict_body_performance(age, gender, height_cm, weight_kg, body_fat_percentage, diastolic, systolic, grip_force, sit_and_bend_forward_cm, sit_ups_counts, broad_jump_cm)
    print("Predicted performance class:", predicted_class)
