import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def diabetes_predictor():
    import pandas as pd 
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder 
    # Load the dataset
    df = pd.read_csv(r'C:\Users\sgoff\Downloads\diabetes_prediction_dataset.csv.zip')

    # Map gender to numerical values
    gender_map = {'Male': 0, 'Female': 1}
    df['gender'] = df['gender'].map(gender_map)

    # Encode smoking history using OneHotEncoder
    One = OneHotEncoder()
    encoded_data = One.fit_transform(df['smoking_history'].values.reshape(-1,1)).toarray()
    feature_names = One.categories_[0]
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names)
    df = pd.concat([df.drop('smoking_history', axis=1), encoded_df], axis=1)
    df = df.dropna(axis=0, how='any')

    # Split features and target variable
    Target = df['diabetes']
    Y = np.array(Target)
    feature = df[['gender','age','hypertension','heart_disease','bmi','HbA1c_level','blood_glucose_level','No Info','current','ever','former','never','not current']]
    X = np.array(feature)

    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=9)
    model = knn.fit(X_train, Y_train)

    # Prompt user for input
    gender_input = input("Enter gender (Male or Female): ").title()
    gender = 1 if gender_input == 'Female' else 0
    age = float(input("Enter age: "))
    hypertension_input = input("Hypertension (Yes or No): ").title()
    hypertension = 1 if hypertension_input == 'Yes' else 0
    heart_disease_input = input("Heart Disease (Yes or No): ").title()
    heart_disease = 1 if heart_disease_input == 'Yes' else 0
    bmi = float(input("Enter BMI: "))
    HbA1c_level = float(input("Enter HbA1c level: "))
    blood_glucose_level = float(input("Enter blood glucose level: "))
    smoking_history = input("Smoking History (No Info, current, ever, former, never, not current): ")

    # Encode smoking history input
    encoded_smoking_history = One.transform([[smoking_history]]).toarray().flatten()

    # Prepare input data for prediction
    input_data = np.array([[gender, age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, *encoded_smoking_history]])

    # Make prediction
    prediction = model.predict(input_data)[0]
    if prediction == 0:
        print("No diabetes risk detected.")
    else:
        print("Diabetes risk detected.")

# Call the function to make predictions based on user input
diabetes_predictor()
