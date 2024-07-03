import pandas as pd 
import numpy as np
from sklearn import preprocessing as pr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as RF

class Diseases:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.dtype = self.data.dtypes

    def transform(self):
        for i in range(self.data.shape[1]):
            if self.data.dtypes[self.data.columns[i]] == 'O':
                label_encoder = pr.LabelEncoder()
                self.data[self.data.columns[i]] = label_encoder.fit_transform(self.data[self.data.columns[i]])
                print(label_encoder.classes_)

        self.x = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]


    def Migraine(self):
        self.transform()  # Call transform before splitting
        columns_to_keep = ['Age', 'Duration', 'Frequency', 'Nausea', 'Phonophobia', 'Photophobia', 'Visual', 'Conscience', 'Paresthesia']
        self.x = self.x[columns_to_keep]  # Select only the desired columns
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, shuffle=False)
        model = RF(n_estimators=5,criterion='entropy')
        model.fit(x_train, y_train)

        # Questions for seizure diagnosis
        questions = [
                "How old are you?",
                "How long does each seizure last (in minutes)?",
                "How often do you experience seizures per month?",
                "Do you experience nausea during seizures? (1 for yes, 0 for no)",
                "Do you have phonophobia (sensitivity to sound) during seizures? (1 for yes, 0 for no)",
                "Do you have photophobia (sensitivity to light) during seizures? (1 for yes, 0 for no)",
                "Do you experience visual disturbances during seizures? (1 for yes, 0 for no)",
                "Do you lose consciousness during seizures? (1 for yes, 0 for no)",
                "Do you experience paresthesia (tingling or numbness) during seizures? (1 for yes, 0 for no)",
            ]
            
        user_responses = []

        # Iterate through questions and store responses
        for question in questions:
            response = input(question + " ")
            user_responses.append(float(response))  # Convert the response to a float
        

        new_data = pd.DataFrame({
                    'Age': [user_responses[0]],
                    'Duration': [user_responses[1]],
                    'Frequency': [user_responses[2]],
                    'Nausea': [user_responses[3]],
                    'Phonophobia': [user_responses[4]],
                    'Photophobia': [user_responses[5]],
                    'Visual': [user_responses[6]],
                    'Conscience': [user_responses[7]],
                    'Paresthesia': [user_responses[8]]
                    
                })
            
        seizure_prediction = model.predict(new_data)


        # Suggested advice based on the predicted outcome
        if seizure_prediction == 0:
            advice = "It seems like you may be experiencing basilar-type aura. Make sure to follow your prescribed medications and avoid triggers such as stress and lack of sleep. It's essential to communicate any changes in symptoms with your healthcare provider."
        elif seizure_prediction == 1:
            advice = "The prediction indicates familial hemiplegic migraine. Exercise caution with medications, follow your doctor's instructions, and report any symptoms promptly. Regularly monitor and discuss your condition with your healthcare provider."
        elif seizure_prediction == 2:
            advice = "It appears to be migraine without aura. Try to avoid known migraine triggers and maintain a healthy lifestyle. If you experience any changes in your symptoms, consult with your healthcare provider for further guidance."
        elif seizure_prediction == 4:
            advice = "The prediction suggests sporadic hemiplegic migraine. Be vigilant with medications, watch for potential interactions, and report any adverse effects. Keep track of your symptoms and communicate regularly with your healthcare provider."
        elif seizure_prediction == 5:
            advice = "You may be experiencing typical aura with migraine. Consider using pain relievers for headaches and avoiding identified triggers. However, it is crucial to consult with your healthcare provider for personalized advice and management."
        elif seizure_prediction == 6:
            advice = "The prediction indicates typical aura without migraine. Keep a record of your symptoms and consult your doctor if you notice any changes. Maintaining a healthy lifestyle and avoiding potential triggers can be beneficial."
        else:
            advice = "The predicted outcome suggests a different type of condition. It is recommended to consult with your doctor for a proper diagnosis and to discuss suitable treatment options."

        # Print the suggested advice
        print(advice)



    def Diabetes(self):
        self.transform()
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, shuffle=False)
        model = RF(n_estimators=5,criterion='entropy')
        model.fit(x_train, y_train)


        # Questions
        questions = [
        "How many pregnancies have you had?",
        "What is your glucose level?",
        "What is your blood pressure?",
        "What is your skin thickness?",
        "What is your insulin level?",
        "What is your BMI (Body Mass Index)?",
        "What is your diabetes pedigree function?",
        "How old are you?"
                        ]
        user_responses = []

        # Iterate through questions and store responses
        for question in questions:
            response = input(question + " ")
            user_responses.append(float(response))  # Convert the response to a float
            
        new_data = pd.DataFrame({
                'Pregnancies': [user_responses[0]],
                'Glucose': [user_responses[1]],
                'BloodPressure': [user_responses[2]],
                'SkinThickness': [user_responses[3]],
                'Insulin': [user_responses[4]],
                'BMI': [user_responses[5]],
                'DiabetesPedigreeFunction': [user_responses[6]],
                'Age': [user_responses[7]]
            })

        y_predict = model.predict(new_data)
        if y_predict[0] == 1:
             print("You are at risk of diabetes. Please consult with a healthcare professional.")
        else:
            print("Your risk of diabetes seems to be low. Maintain a healthy lifestyle for prevention.")
        
    
    def Cancer(self):
        self.transform()  # Call transform before splitting
        columns_to_keep = ['GENDER','AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'ALLERGY ', 
                           'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
        self.x = self.x[columns_to_keep]  # Select only the desired columns
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, shuffle=True,random_state=42)
        model = RF(n_estimators=5,criterion='entropy')
        model.fit(x_train, y_train)

        # Questions for seizure diagnosis
        questions = [
            'Are you Male or Felmale(0 for Female, 1 for Male)'
            ,"How old are you?",
            "Do you smoke?(2 for yes, 1 for no) ",
            "Are your fingers yellow? (2 for yes, 1 for no) ",
            "Do you suffer from anxiety? (2 for yes, 1 for no)",
            "Are there allergies?(2 for yes, 1 for no)",
            "Do you suffer from shortness of breath? (2 for yes, 1 for no)",
            "Do you suffer from difficulty swallowing? (2 for yes, 1 for no)",
            "Do you suffer from chest pain?(2 for yes, 1 for no)"
        ]
        user_responses = []

        # Iterate through questions and store responses
        for question in questions:
            response = input(question + " ")
            user_responses.append(float(response))  # Convert the response to a float

        new_data = pd.DataFrame({
            'GENDER':[user_responses[0]]
            ,'AGE': [user_responses[1]],
            'SMOKING': [user_responses[2]],
            'YELLOW_FINGERS': [user_responses[3]],
            'ANXIETY': [user_responses[4]],
            'ALLERGY ': [user_responses[5]],
            'SHORTNESS OF BREATH': [user_responses[6]],
            'SWALLOWING DIFFICULTY': [user_responses[7]],
            'CHEST PAIN': [user_responses[8]]
        })

        seizure_prediction = model.predict(new_data)

        # Suggested advice based on the predicted outcome
        if seizure_prediction == 0:
            advice = "You do not suffer from cancer, but you must follow up periodically with a specialist doctor"
        elif seizure_prediction == 1:
            advice = "It appears that you are suffering from cancer. You should identify and communicate with the competent health care team, and seeking psychological and social support can be important."
        # Print the suggested advice
        print(advice)

url_Migraine='migraine_data.csv'
url_Diabetes='diabetes.csv'
url_Cancer='survey lung cancer.csv'

if __name__ == '__main__':
    train_Migraine = Diseases(url_Migraine)
    train_diabetes = Diseases(url_Diabetes)
    train_cancer = Diseases(url_Cancer)

    # Ask the patient about the common symptom
    headache_response = input("Do you feel extremely thirsty and constantly dehydrated? (1 for yes, 0 for no) ")
    
    if float(headache_response) == 1:
        # If the patient has a headache, proceed with migraine prediction
        train_diabetes.Diabetes()
    else:
        # If no headache, ask about other symptoms or proceed to other diseases
        print("You do not have a headache. Let's check for other symptoms.")
        # Add additional questions or conditions as needed
        
        # Example: Ask about fatigue
        
        fatigue_response = input("Is there abnormal swelling in any part of the body? (1 for yes, 0 for no) ")
        if float(fatigue_response) == 1:
            # If fatigue is reported, proceed with diabetes prediction
            train_cancer.Cancer()
        else:
            # If no fatigue, proceed with cancer prediction
            train_Migraine.Migraine()







