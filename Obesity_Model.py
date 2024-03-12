def predict_obesity_category():
    import pandas as pd
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    # Load the dataset
    df = pd.read_csv("/content/obesity_data.csv")
    
    # Preprocess the data
    df['Gender'] = df['Gender'].str.lower()
    df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
    x = df[["Age", "Gender", "Height", "Weight", "BMI", "PhysicalActivityLevel"]]
    y = df[["ObesityCategory"]]
    
    # Encode the target variable
    y_label = LabelEncoder()
    y = y_label.fit_transform(y)
    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Train the KNN model
    knn = KNeighborsClassifier(n_neighbors=7)
    model = knn.fit(x_train, y_train)
    
    # Input values from the user
    age = int(input("Enter age: "))
    gender = float(input("Enter gender (0 for Male, 1 for Female): "))
    height = float(input("Enter height (cm): "))
    weight = float(input("Enter weight (kg): "))
    bmi = float(input("Enter BMI: "))
    physical_activity_level = float(input("Enter physical activity level: "))
    
    # Make predictions
    predicted_category = model.predict([[age, gender, height, weight, bmi, physical_activity_level]])
    print("Predicted obesity category:", predicted_category)

# Call the function to predict obesity category
predict_obesity_category()
