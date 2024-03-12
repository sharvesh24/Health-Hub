import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def train_knn_model_and_predict():
    # Load the dataset
    df = pd.read_csv(r'C:\Users\sgoff\Downloads\heart_disease_health_indicators_BRFSS2015.csv.zip')

    # Select features and target
    Features = df[['HighBP', 'HighChol', 'CholCheck', 'BMI',
               'Smoker', 'Stroke', 'Diabetes', 'PhysActivity', 'Fruits', 'Veggies',
               'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
               'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
               'Income']]
    Target = df['HeartDiseaseorAttack']

    # Split the dataset into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(Features, Target, test_size=0.2, random_state=42)

    # Train the K-Nearest Neighbors classifier
    knn = KNeighborsClassifier(n_neighbors=27)
    model_1 = knn.fit(X_train, Y_train)

    # Function to get user input and predict outcome
    def predict_heart_disease_or_attack():
        # Prompt user for feature values
        print("Please provide the following information:")
        HighBP = int(input("High Blood Pressure (0 for No, 1 for Yes): "))
        HighChol = int(input("High Cholesterol (0 for No, 1 for Yes): "))
        CholCheck = int(input("Cholesterol Checked (0 for No, 1 for Yes): "))
        BMI = float(input("Body Mass Index (BMI): "))
        Smoker = int(input("Smoker (0 for No, 1 for Yes): "))
        Stroke = int(input("Stroke (0 for No, 1 for Yes): "))
        Diabetes = int(input("Diabetes (0 for No, 1 for Yes): "))
        PhysActivity = int(input("Physical Activity (0 for No, 1 for Yes): "))
        Fruits = int(input("Daily Fruits Consumption (number of servings): "))
        Veggies = int(input("Daily Veggies Consumption (number of servings): "))
        HvyAlcoholConsump = int(input("Heavy Alcohol Consumption (0 for No, 1 for Yes): "))
        AnyHealthcare = int(input("Any Healthcare (0 for No, 1 for Yes): "))
        NoDocbcCost = int(input("No Doctor Because of Cost (0 for No, 1 for Yes): "))
        GenHlth = int(input("General Health (1-5 scale, 1 being poor, 5 being excellent): "))
        MentHlth = int(input("Mental Health (1-30 scale, 1 being poor, 30 being excellent): "))
        PhysHlth = int(input("Physical Health (1-30 scale, 1 being poor, 30 being excellent): "))
        DiffWalk = int(input("Difficulty Walking (0 for No, 1 for Yes): "))
        Sex = int(input("Sex (0 for Female, 1 for Male): "))
        Age = int(input("Age: "))
        Education = int(input("Education (1-6 scale, 1 being less than high school, 6 being college graduate or above): "))
        Income = int(input("Income (1-8 scale, 1 being less than $15,000, 8 being $75,000 or more): "))

        # Prepare input data
        input_data = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, Diabetes, PhysActivity,
                                Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth,
                                MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income]])

        # Make prediction
        predicted_value = model_1.predict(input_data)

        # Print prediction
        if predicted_value == 1:
            print("Predicted outcome: Likely to have heart disease or attack.")
        else:
            print("Predicted outcome: Unlikely to have heart disease or attack.")

    # Call the function to make prediction based on user input
    predict_heart_disease_or_attack()

# Call the function to train the model and make prediction
train_knn_model_and_predict()
