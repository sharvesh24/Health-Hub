import tkinter as Health_hub
import tkinter.ttk as ttk
from tkinter import *
from tkinter.ttk import *
from tkinter.ttk import Label, Style
from PIL import ImageTk, Image
from tkinter import scrolledtext
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

window_main=Tk()
window_main.title("Health Hub")
window_main.resizable(False,False)
Health_hub.Label()
ttk.Label()
Label()
Text()

window_main.configure(bg='gray35', height=1028, width=2056)
window_main.geometry("2056x1028")

Label_1 = Health_hub.Label(window_main, text='Health Hub', foreground='skyblue', bg='gray35', font='Algerian 80', borderwidth='20', relief='groove')
Label_1.place(relx=0.5, rely=0.3, anchor='center')

Label_2 = Health_hub.Label(window_main, text='THE GREATEST WEALTH IS HEALTH', foreground='lightgoldenrod', bg='gray35', font='century 50')
Label_2.place(relx=0.5, rely=0.1, anchor='center')

Label_3 = Health_hub.Label(window_main, text='Positive thinking will let you do everything better than negative thinking will', foreground='orange', bg='gray35', font='forte 25')
Label_3.place(relx=0.5, rely=0.8, anchor='center')

Label_4 = Health_hub.Label(window_main, text="You've got to stay strong to be strong in tough times", foreground='yellow', bg='gray35', font='forte 25')
Label_4.place(relx=0.5, rely=0.88, anchor='center')

img=Image.open("D://Programs//HackNova//Health Hub//Health.jpeg").resize((350,250))
test=ImageTk.PhotoImage(img)
lbl=Health_hub.Label(window_main,image=test)
lbl.image=test
lbl.place(relx=0.15, rely=0.385, anchor='center')

img2=Image.open("D://Programs//HackNova//Health Hub//images.jpeg").resize((350,250))
test=ImageTk.PhotoImage(img2)
lbl1=Health_hub.Label(window_main,image=test)
lbl1.image=test
lbl1.place(relx=0.855, rely=0.385, anchor='center')

def openNewWindow():
    window = Toplevel(window_main)
    window.title("Prediction")
    window.state("zoomed")
    window.resizable(False, False)
    window.configure(bg='gray35', height=1028, width=2056)
    window.geometry("2056x1028")
    Prediction_txt = Health_hub.Label(window, text="Prediction", foreground='salmon', background='gray35', font='algerian 50')
    Prediction_txt.place(relx=0.5, rely=0.1, anchor='center')

    img3=Image.open("D://Programs//HackNova//Health Hub//images(1).jpeg").resize((450,250))
    test=ImageTk.PhotoImage(img3)
    lbl3=Health_hub.Label(window,image=test)
    lbl3.image=test
    lbl3.place(relx=0.75, rely=0.35, anchor='center')

    img4=Image.open("D://Programs//HackNova//Health Hub//download (1).png").resize((450,250))
    test=ImageTk.PhotoImage(img4)
    lbl4=Health_hub.Label(window,image=test)
    lbl4.image=test
    lbl4.place(relx=0.75, rely=0.75, anchor='center')

    def openNewWindow():
        new_window=Toplevel(window)
        new_window.title("Diabetes")
        new_window.state("zoomed")
        new_window.resizable(False, False)
        new_window.configure(bg='gray35', heigh=1028, width=2056)
        new_window.geometry("2056x1028")
        diabetes_txt = Health_hub.Label(new_window, text="Diabetes", foreground='salmon', background='gray35', font='algerian 50')
        diabetes_txt.place(relx=0.5, rely=0.1, anchor='center')

        l1 = Health_hub.Label(new_window, text="Enter Gender: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l1.place(relx=0.3,rely=0.2, anchor='center')
        gender = Health_hub.Entry(new_window, bd=5, width=20)
        gender.place(relx=0.55, rely=0.2, anchor='center')

        l2 = Health_hub.Label(new_window, text="Enter Age: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l2.place(relx=0.3,rely=0.25, anchor='center')
        age = Health_hub.Entry(new_window, bd=5, width=20)
        age.place(relx=0.55, rely=0.25, anchor='center')
       
            
        l3 = Health_hub.Label(new_window, text="Hypertension (Yes or No): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l3.place(relx=0.3,rely=0.3, anchor='center')
        hypertension = Health_hub.Entry(new_window, bd=5, width=20)
        hypertension.place(relx=0.55, rely=0.3, anchor='center')
        
            
        l4 = Health_hub.Label(new_window, text="Heart Disease (Yes or No): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l4.place(relx=0.3,rely=0.35, anchor='center')
        heart_disease = Health_hub.Entry(new_window, bd=5, width=20)
        heart_disease.place(relx=0.55, rely=0.35, anchor='center')

        l5 = Health_hub.Label(new_window, text="Enter BMI: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l5.place(relx=0.3,rely=0.4, anchor='center')
        bmi = Health_hub.Entry(new_window, bd=5, width=20)
        bmi.place(relx=0.55, rely=0.4, anchor='center')
       
            
        l6 = Health_hub.Label(new_window, text="Enter HbA1c level:  ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l6.place(relx=0.3,rely=0.45, anchor='center')
        HbA1c_level = Health_hub.Entry(new_window, bd=5, width=20)
        HbA1c_level.place(relx=0.55, rely=0.45, anchor='center')
        
            
        l7 = Health_hub.Label(new_window, text="Enter blood glucose level: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l7.place(relx=0.3,rely=0.5, anchor='center')
        blood_glucose_level = Health_hub.Entry(new_window, bd=5, width=20)
        blood_glucose_level.place(relx=0.55, rely=0.5, anchor='center')

        l8 = Health_hub.Label(new_window, text="Smoking History (No Info, current, ever, former, never, not current): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l8.place(relx=0.3,rely=0.55, anchor='center')
        smoking_history = Health_hub.Entry(new_window, bd=5, width=20)
        smoking_history.place(relx=0.55, rely=0.55, anchor='center')

        def diabetes():
            df = pd.read_csv(r'diabetes_prediction_dataset.csv')

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


            # Encode smoking history input
            encoded_smoking_history = One.transform([[smoking_history]]).toarray().flatten()

            # Prepare input data for prediction
            input_data = np.array([[gender, age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, *encoded_smoking_history]])

            # Make prediction
            prediction = model.predict(input_data)[0]

            l8 = Health_hub.Label(new_window, text=prediction, font=("Arial",15), background='gray35', foreground='floralwhite')
            l8.place(relx=0.475,rely=0.75, anchor='center')
         
        btn = Health_hub.Button(new_window, text="Enter", foreground='salmon', bg='gray35', font='calibiri 25', command=diabetes)
        btn.place(relx=0.475, rely=0.65, anchor='center')
        btn.place(x=0, y=20)

    btn = Health_hub.Button(window, text="> Diabetes", command=openNewWindow, foreground='floralwhite', bg='gray35', font='calibiri 25')
    btn.place(relx=0.111, rely=0.3, anchor='center')
    btn.place(x=0, y=20)

    def openNewWindow():
        new_window=Toplevel(window)
        new_window.title("Overall Body Performance")
        new_window.state("zoomed")
        new_window.resizable(False, False)
        new_window.configure(bg='gray35', heigh=1028, width=2056)
        new_window.geometry("2056x1028")
        obp_txt = Health_hub.Label(new_window, text="Overall Body Performance", foreground='salmon', background='gray35', font='algerian 50')
        obp_txt.place(relx=0.5, rely=0.1, anchor='center')

        l1 = Health_hub.Label(new_window, text="Enter Gender: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l1.place(relx=0.3,rely=0.2, anchor='center')
        gender = Health_hub.Entry(new_window, bd=5, width=20)
        gender.place(relx=0.55, rely=0.2, anchor='center')

        l2 = Health_hub.Label(new_window, text="Enter Age: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l2.place(relx=0.3,rely=0.25, anchor='center')
        age = Health_hub.Entry(new_window, bd=5, width=20)
        age.place(relx=0.55, rely=0.25, anchor='center')
       
            
        l3 = Health_hub.Label(new_window, text="Enter height in cm: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l3.place(relx=0.3,rely=0.3, anchor='center')
        height_cm = Health_hub.Entry(new_window, bd=5, width=20)
        height_cm.place(relx=0.55, rely=0.3, anchor='center')
        
            
        l4 = Health_hub.Label(new_window, text="Enter weight in kg: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l4.place(relx=0.3,rely=0.35, anchor='center')
        weight_kg = Health_hub.Entry(new_window, bd=5, width=20)
        weight_kg.place(relx=0.55, rely=0.35, anchor='center')

        l5 = Health_hub.Label(new_window, text="Enter body fat percentage: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l5.place(relx=0.3,rely=0.4, anchor='center')
        body_fat_percentage = Health_hub.Entry(new_window, bd=5, width=20)
        body_fat_percentage.place(relx=0.55, rely=0.4, anchor='center')
       
            
        l6 = Health_hub.Label(new_window, text="Enter diastolic blood pressure: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l6.place(relx=0.3,rely=0.45, anchor='center')
        diastolic = Health_hub.Entry(new_window, bd=5, width=20)
        diastolic.place(relx=0.55, rely=0.45, anchor='center')
        
            
        l7 = Health_hub.Label(new_window, text="Enter systolic blood pressure: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l7.place(relx=0.3,rely=0.5, anchor='center')
        systolic = Health_hub.Entry(new_window, bd=5, width=20)
        systolic.place(relx=0.55, rely=0.5, anchor='center')

        l8 = Health_hub.Label(new_window, text="Enter grip force: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l8.place(relx=0.3,rely=0.55, anchor='center')
        grip_force = Health_hub.Entry(new_window, bd=5, width=20)
        grip_force.place(relx=0.55, rely=0.55, anchor='center')

        l9 = Health_hub.Label(new_window, text="Enter sit and bend forward measurement in cm: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l9.place(relx=0.3,rely=0.6, anchor='center')
        sit_and_bend_forward_cm = Health_hub.Entry(new_window, bd=5, width=20)
        sit_and_bend_forward_cm.place(relx=0.55, rely=0.6, anchor='center')
        
            
        l10 = Health_hub.Label(new_window, text="Enter sit-ups counts: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l10.place(relx=0.3,rely=0.65, anchor='center')
        sit_ups_counts = Health_hub.Entry(new_window, bd=5, width=20)
        sit_ups_counts.place(relx=0.55, rely=0.65, anchor='center')

        l11 = Health_hub.Label(new_window, text="Enter broad jump distance in cm: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l11.place(relx=0.3,rely=0.7, anchor='center')
        broad_jump_cm = Health_hub.Entry(new_window, bd=5, width=20)
        broad_jump_cm.place(relx=0.55, rely=0.7, anchor='center')

        def predict_body_performance():
            df = pd.read_csv(r'bodyPerformance.csv')

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
            l = Health_hub.Label(new_window, text=predicted_class, font=("Arial",15), background='gray35', foreground='floralwhite')
            l.place(relx=0.5,rely=0.9, anchor='center')

            return predicted_class
        
         
        btn = Health_hub.Button(new_window, text="Enter", foreground='salmon', bg='gray35', font='calibiri 25', command=predict_body_performance)
        btn.place(relx=0.5, rely=0.8, anchor='center')
        btn.place(x=0, y=20)

    btn = Health_hub.Button(window, text="> Overall Body Performance", command=openNewWindow, foreground='floralwhite', bg='gray35', font='calibiri 25')
    btn.place(relx=0.19, rely=0.4, anchor='center')
    btn.place(x=0, y=20)
        
    

    def openNewWindow():
        new_window=Toplevel(window)
        new_window.title("Heart Disease")
        new_window.state("zoomed")
        new_window.resizable(False, False)
        new_window.configure(bg='gray35', heigh=1028, width=2056)
        new_window.geometry("2056x1028")
        heart_txt = Health_hub.Label(new_window, text="Heart Disease", foreground='salmon', background='gray35', font='algerian 50')
        heart_txt.place(relx=0.5, rely=0.1, anchor='center')

        l1 = Health_hub.Label(new_window, text="High Blood Pressure (0 for No, 1 for Yes): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l1.place(relx=0.15,rely=0.2, anchor='center')
        HighBP = Health_hub.Entry(new_window, bd=5, width=20)
        HighBP.place(relx=0.35, rely=0.2, anchor='center')

        l2 = Health_hub.Label(new_window, text="High Cholesterol (0 for No, 1 for Yes): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l2.place(relx=0.15,rely=0.25, anchor='center')
        HighChol = Health_hub.Entry(new_window, bd=5, width=20)
        HighChol.place(relx=0.35, rely=0.25, anchor='center')
       
            
        l3 = Health_hub.Label(new_window, text="Cholesterol Checked (0 for No, 1 for Yes): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l3.place(relx=0.15,rely=0.3, anchor='center')
        CholCheck = Health_hub.Entry(new_window, bd=5, width=20)
        CholCheck.place(relx=0.35, rely=0.3, anchor='center')
        
            
        l4 = Health_hub.Label(new_window, text="Body Mass Index (BMI): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l4.place(relx=0.15,rely=0.35, anchor='center')
        BMI = Health_hub.Entry(new_window, bd=5, width=20)
        BMI.place(relx=0.35, rely=0.35, anchor='center')

        l5 = Health_hub.Label(new_window, text="Smoker (0 for No, 1 for Yes): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l5.place(relx=0.15,rely=0.4, anchor='center')
        Smoker = Health_hub.Entry(new_window, bd=5, width=20)
        Smoker.place(relx=0.35, rely=0.4, anchor='center')
       
            
        l6 = Health_hub.Label(new_window, text="Stroke (0 for No, 1 for Yes): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l6.place(relx=0.15,rely=0.45, anchor='center')
        Stroke = Health_hub.Entry(new_window, bd=5, width=20)
        Stroke.place(relx=0.35, rely=0.45, anchor='center')
        
            
        l7 = Health_hub.Label(new_window, text="Diabetes (0 for No, 1 for Yes): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l7.place(relx=0.15,rely=0.5, anchor='center')
        Diabetes = Health_hub.Entry(new_window, bd=5, width=20)
        Diabetes.place(relx=0.35, rely=0.5, anchor='center')

        l8 = Health_hub.Label(new_window, text="Daily Fruits Consumption (number of servings): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l8.place(relx=0.15,rely=0.55, anchor='center')
        Fruits = Health_hub.Entry(new_window, bd=5, width=20)
        Fruits.place(relx=0.35, rely=0.55, anchor='center')

        l9 = Health_hub.Label(new_window, text="Physical Activity (0 for No, 1 for Yes): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l9.place(relx=0.15,rely=0.6, anchor='center')
        PhysActivity = Health_hub.Entry(new_window, bd=5, width=20)
        PhysActivity.place(relx=0.35, rely=0.6, anchor='center')
        
            
        l10 = Health_hub.Label(new_window, text="Daily Veggies Consumption (number of servings): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l10.place(relx=0.15,rely=0.65, anchor='center')
        Veggies = Health_hub.Entry(new_window, bd=5, width=20)
        Veggies.place(relx=0.35, rely=0.65, anchor='center')

        l11 = Health_hub.Label(new_window, text="Heavy Alcohol Consumption (0 for No, 1 for Yes): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l11.place(relx=0.15,rely=0.7, anchor='center')
        HvyAlcoholConsump = Health_hub.Entry(new_window, bd=5, width=20)
        HvyAlcoholConsump.place(relx=0.35, rely=0.7, anchor='center')

        l12 = Health_hub.Label(new_window, text="Any Healthcare (0 for No, 1 for Yes): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l12.place(relx=0.61,rely=0.2, anchor='center')
        AnyHealthcare = Health_hub.Entry(new_window, bd=5, width=20)
        AnyHealthcare.place(relx=0.81, rely=0.2, anchor='center')

        l13 = Health_hub.Label(new_window, text="No Doctor Because of Cost (0 for No, 1 for Yes): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l13.place(relx=0.61,rely=0.25, anchor='center')
        NoDocbcCost = Health_hub.Entry(new_window, bd=5, width=20)
        NoDocbcCost.place(relx=0.81, rely=0.25, anchor='center')
       
            
        l14 = Health_hub.Label(new_window, text="General Health (1-5 scale): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l14.place(relx=0.61,rely=0.3, anchor='center')
        GenHlth = Health_hub.Entry(new_window, bd=5, width=20)
        GenHlth.place(relx=0.81, rely=0.3, anchor='center')
        
            
        l15 = Health_hub.Label(new_window, text="Physical Health (1-30 scale): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l15.place(relx=0.61,rely=0.35, anchor='center')
        PhysHlth = Health_hub.Entry(new_window, bd=5, width=20)
        PhysHlth.place(relx=0.81, rely=0.35, anchor='center')

        l16 = Health_hub.Label(new_window, text="Mental Health (1-30 scale): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l16.place(relx=0.61,rely=0.4, anchor='center')
        MentHlth = Health_hub.Entry(new_window, bd=5, width=20)
        MentHlth.place(relx=0.81, rely=0.4, anchor='center')
       
            
        l17= Health_hub.Label(new_window, text="Difficulty Walking (0 for No, 1 for Yes): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l17.place(relx=0.61,rely=0.45, anchor='center')
        DiffWalk = Health_hub.Entry(new_window, bd=5, width=20)
        DiffWalk.place(relx=0.81, rely=0.45, anchor='center')
        
            
        l18 = Health_hub.Label(new_window, text="Sex (0 for Female, 1 for Male): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l18.place(relx=0.61,rely=0.5, anchor='center')
        Sex = Health_hub.Entry(new_window, bd=5, width=20)
        Sex.place(relx=0.81, rely=0.5, anchor='center')

        l19 = Health_hub.Label(new_window, text="Age: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l19.place(relx=0.61,rely=0.55, anchor='center')
        Age = Health_hub.Entry(new_window, bd=5, width=20)
        Age.place(relx=0.81, rely=0.55, anchor='center')

        l20 = Health_hub.Label(new_window, text="Education (1-6 scale): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l20.place(relx=0.61,rely=0.6, anchor='center')
        Education = Health_hub.Entry(new_window, bd=5, width=20)
        Education.place(relx=0.81, rely=0.6, anchor='center')

            
        l21 = Health_hub.Label(new_window, text="Income (1-8 scale): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l21.place(relx=0.61,rely=0.65, anchor='center')
        Income = Health_hub.Entry(new_window, bd=5, width=20)
        Income.place(relx=0.81, rely=0.65, anchor='center')


        def train_knn_model_and_predict():
            # Load the dataset
            df = pd.read_csv(r'heart_disease_health_indicators_BRFSS2015.csv')

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
                
                # Prepare input data
                input_data = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, Diabetes, PhysActivity,
                                        Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth,
                                        MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income]])

                # Make prediction
                predicted_value = model_1.predict(input_data)

                # Print prediction
                if predicted_value == 1:
                    l = Health_hub.Label(new_window, text='Predicted outcome: Likely to have heart disease or attack', font=("Arial",15), background='gray35', foreground='floralwhite')
                    l.place(relx=0.5,rely=0.9, anchor='center')
                else:
                    l = Health_hub.Label(new_window, text='Predicted outcome: Unlikely to have heart disease or attack.', font=("Arial",15), background='gray35', foreground='floralwhite')
                    l.place(relx=0.5,rely=0.9, anchor='center')
                  
            # Call the function to make prediction based on user input
            predict_heart_disease_or_attack()

        # Call the function to train the model and make prediction
            
         
        btn = Health_hub.Button(new_window, text="Enter", foreground='salmon', bg='gray35', font='calibiri 25', command=train_knn_model_and_predict)
        btn.place(relx=0.5, rely=0.8, anchor='center')
        btn.place(x=0, y=20)

    btn = Health_hub.Button(window, text="> Heart Disease", command=openNewWindow, foreground='floralwhite', bg='gray35', font='calibiri 25')
    btn.place(relx=0.135, rely=0.5, anchor='center')
    btn.place(x=0, y=20)
    

    def openNewWindow():
        new_window=Toplevel(window)
        new_window.title("Obesity")
        new_window.state("zoomed")
        new_window.resizable(False, False)
        new_window.configure(bg='gray35', heigh=1028, width=2056)
        new_window.geometry("2056x1028")
        obesity_txt = Health_hub.Label(new_window, text="Obesity", foreground='salmon', background='gray35', font='algerian 50')
        obesity_txt.place(relx=0.5, rely=0.1, anchor='center')

        l1 = Health_hub.Label(new_window, text="Enter age: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l1.place(relx=0.3,rely=0.2, anchor='center')
        age = Health_hub.Entry(new_window, bd=5, width=20)
        age.place(relx=0.55, rely=0.2, anchor='center')

        l2 = Health_hub.Label(new_window, text="Enter gender (0 for Male, 1 for Female): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l2.place(relx=0.3,rely=0.25, anchor='center')
        gender = Health_hub.Entry(new_window, bd=5, width=20)
        gender.place(relx=0.55, rely=0.25, anchor='center')
       
            
        l3 = Health_hub.Label(new_window, text="Enter height (cm): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l3.place(relx=0.3,rely=0.3, anchor='center')
        height = Health_hub.Entry(new_window, bd=5, width=20)
        height.place(relx=0.55, rely=0.3, anchor='center')
        
            
        l4 = Health_hub.Label(new_window, text="Enter weight (kg): ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l4.place(relx=0.3,rely=0.35, anchor='center')
        weight = Health_hub.Entry(new_window, bd=5, width=20)
        weight.place(relx=0.55, rely=0.35, anchor='center')

        l5 = Health_hub.Label(new_window, text="Enter BMI: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l5.place(relx=0.3,rely=0.4, anchor='center')
        bmi = Health_hub.Entry(new_window, bd=5, width=20)
        bmi.place(relx=0.55, rely=0.4, anchor='center')
       
            
        l6 = Health_hub.Label(new_window, text="Enter physical activity level: ", font=("Arial",15), background='gray35', foreground='floralwhite')
        l6.place(relx=0.3,rely=0.45, anchor='center')
        physical_activity_level = Health_hub.Entry(new_window, bd=5, width=20)
        physical_activity_level.place(relx=0.55, rely=0.45, anchor='center')

        def predict_obesity_category():
         # Load the dataset
         df = pd.read_csv("obesity_data.csv")
         
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
         
         # Make predictions
         predicted_category = model.predict([[age, gender, height, weight, bmi, physical_activity_level]])
         print("Predicted obesity category:", predicted_category)
         l = Health_hub.Label(new_window, text=predicted_category, font=("Arial",15), background='gray35', foreground='floralwhite')
         l.place(relx=0.4,rely=0.65, anchor='center')
         
        btn = Health_hub.Button(new_window, text="Enter", foreground='salmon', bg='gray35', font='calibiri 25', command=predict_obesity_category)
        btn.place(relx=0.4, rely=0.55, anchor='center')
        btn.place(x=0, y=20)
   
    btn = Health_hub.Button(window, text="> Obesity", command=openNewWindow, foreground='floralwhite', bg='gray35', font='calibiri 25')
    btn.place(relx=0.106, rely=0.6, anchor='center')
    btn.place(x=0, y=20)
    

btn = Health_hub.Button(window_main, text="Prediction", command=openNewWindow, foreground='salmon', bg='gray35', font='calibiri 25')
btn.place(relx=0.3, rely=0.6, anchor='center')
btn.place(x=0, y=20)
    
def openNewWindow():
    window = Toplevel(window_main)
    window.title("Learn")
    window.state("zoomed")
    window.resizable(False, False)
    window.configure(bg='gray35', height=1028, width=2056)
    window.geometry("2056x1028")
    Learn_txt = Health_hub.Label(window, text="Learn", foreground='salmon', background='gray35', font='algerian 50')
    Learn_txt.place(relx=0.5, rely=0.1, anchor='center')

    img2=Image.open("D://Programs//HackNova//Health Hub//download.jpeg").resize((350,250))
    test=ImageTk.PhotoImage(img2)
    lbl1=Health_hub.Label(window,image=test)
    lbl1.image=test
    lbl1.place(relx=0.855, rely=0.35, anchor='center')

    def openNewWindow():
        new_window=Toplevel(window)
        new_window.title("Diabetes")
        new_window.state("zoomed")
        new_window.resizable(False, False)
        new_window.configure(bg='gray35', heigh=1028, width=2056)
        new_window.geometry("2056x1028")
        diabetes_txt = Health_hub.Label(new_window, text="Diabetes", foreground='salmon', background='gray35', font='algerian 35')
        diabetes_txt.place(relx=0.5, rely=0.08, anchor='center')

        dia_txt1 = Health_hub.Label(new_window, text="Diabetes, a chronic metabolic disorder characterized by elevated blood sugar levels, presents a ", foreground='floralwhite', background='gray35', font='calibiri 25')
        dia_txt1.place(relx=0.5, rely=0.15, anchor='center')
        dia_txt2 = Health_hub.Label(new_window, text="significant public health challenge globally. Type 1 diabetes results from the immune system", foreground='floralwhite', background='gray35', font='calibiri 25')
        dia_txt2.place(relx=0.5, rely=0.2, anchor='center')
        dia_txt3 = Health_hub.Label(new_window, text="attacking insulin-producing beta cells in the pancreas, while type 2 diabetes arises from insulin ", foreground='floralwhite', background='gray35', font='calibiri 25')
        dia_txt3.place(relx=0.5, rely=0.25, anchor='center')
        dia_txt4 = Health_hub.Label(new_window, text="resistance and impaired insulin secretion. Both types can lead to serious complications, including", foreground='floralwhite', background='gray35', font='calibiri 25')
        dia_txt4.place(relx=0.5, rely=0.3, anchor='center')
        dia_txt5 = Health_hub.Label(new_window, text="cardiovascular disease, kidney failure, blindness, and neuropathy. Effective management ", foreground='floralwhite', background='gray35', font='calibiri 25')
        dia_txt5.place(relx=0.5, rely=0.35, anchor='center')
        dia_txt6 = Health_hub.Label(new_window, text="strategies involve blood sugar monitoring, medication (such as insulin therapy or oral ", foreground='floralwhite', background='gray35', font='calibiri 25')
        dia_txt6.place(relx=0.5, rely=0.4, anchor='center')
        dia_txt7 = Health_hub.Label(new_window, text="hypoglycemic agents), lifestyle modifications (including a balanced diet, regular exercise, and", foreground='floralwhite', background='gray35', font='calibiri 25')
        dia_txt7.place(relx=0.5, rely=0.45, anchor='center')
        dia_txt8 = Health_hub.Label(new_window, text="weight management), and education on self-care practices", foreground='floralwhite', background='gray35', font='calibiri 25')
        dia_txt8.place(relx=0.5, rely=0.5, anchor='center')

        img=Image.open("D://Programs//HackNova//Health Hub//diabetes.jpg").resize((400,250))
        test=ImageTk.PhotoImage(img)
        lbl=Health_hub.Label(new_window,image=test)
        lbl.image=test
        lbl.place(relx=0.5, rely=0.8, anchor='center')


    btn = Health_hub.Button(window, text="> Diabetes", command=openNewWindow, foreground='floralwhite', bg='gray35', font='calibiri 25')
    btn.place(relx=0.111, rely=0.3, anchor='center')
    btn.place(x=0, y=20)

    def openNewWindow():
        new_window=Toplevel(window)
        new_window.title("Overall Body Performance")
        new_window.state("zoomed")
        new_window.resizable(False, False)
        new_window.configure(bg='gray35', heigh=1028, width=2056)
        new_window.geometry("2056x1028")
        obp_txt = Health_hub.Label(new_window, text="Overall Body Performance", foreground='salmon', background='gray35', font='algerian 35')
        obp_txt.place(relx=0.5, rely=0.1, anchor='center')

        obp_txt1 = Health_hub.Label(new_window, text="Optimizing overall body performance involves a holistic approach encompassing physical fitness,  ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obp_txt1.place(relx=0.5, rely=0.15, anchor='center')
        obp_txt2 = Health_hub.Label(new_window, text="mental well-being, and lifestyle habits. Regular exercise, comprising both cardiovascular and  ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obp_txt2.place(relx=0.5, rely=0.2, anchor='center')
        obp_txt3 = Health_hub.Label(new_window, text="strength-training activities, enhances cardiovascular health, muscular strength, flexibility, and ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obp_txt3.place(relx=0.5, rely=0.25, anchor='center')
        obp_txt4 = Health_hub.Label(new_window, text="endurance, contributing to improved overall fitness levels. Adequate nutrition, hydration, and rest ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obp_txt4.place(relx=0.5, rely=0.3, anchor='center')
        obp_txt5 = Health_hub.Label(new_window, text="are essential for supporting exercise routines and facilitating muscle recovery and growth. ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obp_txt5.place(relx=0.5, rely=0.35, anchor='center')
        obp_txt6 = Health_hub.Label(new_window, text="Additionally, prioritizing mental health through stress management techniques, mindfulness ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obp_txt6.place(relx=0.5, rely=0.4, anchor='center')
        obp_txt7 = Health_hub.Label(new_window, text="practices, and sufficient sleep fosters cognitive function, emotional resilience, and overall well- ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obp_txt7.place(relx=0.5, rely=0.45, anchor='center')
        obp_txt8 = Health_hub.Label(new_window, text="being. Balancing these elements promotes optimal body performance, enabling individuals to ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obp_txt8.place(relx=0.5, rely=0.5, anchor='center')
        obp_txt9 = Health_hub.Label(new_window, text="function at their best and maintain a high quality of life. ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obp_txt9.place(relx=0.5, rely=0.55, anchor='center')

        img=Image.open("D://Programs//HackNova//Health Hub//Overall_preformance.jpg").resize((400,250))
        test=ImageTk.PhotoImage(img)
        lbl=Health_hub.Label(new_window,image=test)
        lbl.image=test
        lbl.place(relx=0.5, rely=0.8, anchor='center')

    btn = Health_hub.Button(window, text="> Overall Body Performance", command=openNewWindow, foreground='floralwhite', bg='gray35', font='calibiri 25')
    btn.place(relx=0.19, rely=0.4, anchor='center')
    btn.place(x=0, y=20)

    def openNewWindow():
        new_window=Toplevel(window)
        new_window.title("Heart Disease")
        new_window.state("zoomed")
        new_window.resizable(False, False)
        new_window.configure(bg='gray35', heigh=1028, width=2056)
        new_window.geometry("2056x1028")
        heart_txt = Health_hub.Label(new_window, text="Heart Disease", foreground='salmon', background='gray35', font='algerian 35')
        heart_txt.place(relx=0.5, rely=0.08, anchor='center')

        heart_txt1 = Health_hub.Label(new_window, text="Heart disease encompasses a spectrum of conditions affecting the heart's structure and function, ", foreground='floralwhite', background='gray35', font='calibiri 25')
        heart_txt1.place(relx=0.5, rely=0.15, anchor='center')
        heart_txt2 = Health_hub.Label(new_window, text="constituting a leading cause of morbidity and mortality worldwide. Coronary artery disease (CAD),  ", foreground='floralwhite', background='gray35', font='calibiri 25')
        heart_txt2.place(relx=0.5, rely=0.2, anchor='center')
        heart_txt3 = Health_hub.Label(new_window, text="the most prevalent form, arises from the accumulation of plaque in the coronary arteries, leading ", foreground='floralwhite', background='gray35', font='calibiri 25')
        heart_txt3.place(relx=0.5, rely=0.25, anchor='center')
        heart_txt4 = Health_hub.Label(new_window, text="to reduced blood flow and oxygen supply to the heart muscle. This can manifest as angina (chest ", foreground='floralwhite', background='gray35', font='calibiri 25')
        heart_txt4.place(relx=0.5, rely=0.3, anchor='center')
        heart_txt5 = Health_hub.Label(new_window, text="pain) or result in a heart attack if a plaque ruptures, causing a complete blockage. Heart failure, ", foreground='floralwhite', background='gray35', font='calibiri 25')
        heart_txt5.place(relx=0.5, rely=0.35, anchor='center')
        heart_txt6 = Health_hub.Label(new_window, text="another significant condition, occurs when the heart's pumping ability becomes impaired, often  ", foreground='floralwhite', background='gray35', font='calibiri 25')
        heart_txt6.place(relx=0.5, rely=0.4, anchor='center')
        heart_txt7 = Health_hub.Label(new_window, text="due to longstanding conditions like CAD or hypertension. Symptoms include fatigue, shortness of ", foreground='floralwhite', background='gray35', font='calibiri 25')
        heart_txt7.place(relx=0.5, rely=0.45, anchor='center')
        heart_txt8 = Health_hub.Label(new_window, text="breath, and fluid retention, impacting daily activities and quality of life. ", foreground='floralwhite', background='gray35', font='calibiri 25')
        heart_txt8.place(relx=0.5, rely=0.5, anchor='center')

        img=Image.open("D://Programs//HackNova//Health Hub//Heart disease.jpg").resize((400,250))
        test=ImageTk.PhotoImage(img)
        lbl=Health_hub.Label(new_window,image=test)
        lbl.image=test
        lbl.place(relx=0.5, rely=0.8, anchor='center')


    btn = Health_hub.Button(window, text="> Heart Disease", command=openNewWindow, foreground='floralwhite', bg='gray35', font='calibiri 25')
    btn.place(relx=0.135, rely=0.5, anchor='center')
    btn.place(x=0, y=20)

    def openNewWindow():
        new_window=Toplevel(window)
        new_window.title("Obesity")
        new_window.state("zoomed")
        new_window.resizable(False, False)
        new_window.configure(bg='gray35', heigh=1028, width=2056)
        new_window.geometry("2056x1028")
        obesity_txt = Health_hub.Label(new_window, text="Obesity", foreground='salmon', background='gray35', font='algerian 35')
        obesity_txt.place(relx=0.5, rely=0.05, anchor='center')

        obs_txt1 = Health_hub.Label(new_window, text="Obesity, characterized by an excess accumulation of body fat, presents a significant global health ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obs_txt1.place(relx=0.5, rely=0.15, anchor='center')
        obs_txt2 = Health_hub.Label(new_window, text="challenge driven by unhealthy dietary habits, sedentary lifestyles, and genetic predispositions. Its ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obs_txt2.place(relx=0.5, rely=0.2, anchor='center')
        obs_txt3 = Health_hub.Label(new_window, text="adverse health effects, including increased risks of chronic diseases like diabetes, cardiovascular ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obs_txt3.place(relx=0.5, rely=0.25, anchor='center')
        obs_txt4 = Health_hub.Label(new_window, text="issues, and psychological disorders, underscore the urgent need for multifaceted prevention and ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obs_txt4.place(relx=0.5, rely=0.3, anchor='center')
        obs_txt5 = Health_hub.Label(new_window, text="management strategies. Promoting healthier eating patterns, encouraging regular physical ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obs_txt5.place(relx=0.5, rely=0.35, anchor='center')
        obs_txt6 = Health_hub.Label(new_window, text="activity, and creating supportive environments for lifestyle changes are pivotal in addressing ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obs_txt6.place(relx=0.5, rely=0.4, anchor='center')
        obs_txt7 = Health_hub.Label(new_window, text="obesity on individual and societal levels, alongside public health initiatives aimed at raising ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obs_txt7.place(relx=0.5, rely=0.45, anchor='center')
        obs_txt8 = Health_hub.Label(new_window, text="awareness and implementing policies conducive to healthier living. ", foreground='floralwhite', background='gray35', font='calibiri 25')
        obs_txt8.place(relx=0.5, rely=0.5, anchor='center')

        img=Image.open("D://Programs//HackNova//Health Hub//Obesity.jpg").resize((400,250))
        test=ImageTk.PhotoImage(img)
        lbl=Health_hub.Label(new_window,image=test)
        lbl.image=test
        lbl.place(relx=0.5, rely=0.76, anchor='center')

    btn = Health_hub.Button(window, text="> Obesity", command=openNewWindow, foreground='floralwhite', bg='gray35', font='calibiri 25')
    btn.place(relx=0.106, rely=0.6, anchor='center')
    btn.place(x=0, y=20)

btn = Health_hub.Button(window_main, text="Learn", command=openNewWindow, foreground='salmon', bg='gray35', font='calibiri 25')
btn.place(relx=0.7, rely=0.6, anchor='center')
btn.place(x=0, y=20)

window_main.state('zoomed')
window_main.mainloop()
