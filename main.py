# Importing Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the Data
data = pd.read_csv('insurance_data.csv')

# Data Preprocessing
data['sex'] = data['sex'].apply(lambda x: 1 if x == 'male' else 0)
data['smoker'] = data['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
X = data.drop('insuranceclaim', axis=1)
y = data['insuranceclaim']

# Splitting the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Prediction
age = float(input("Enter age: "))
sex = float(input("Enter sex (0 for female, 1 for male): "))
bmi = float(input("Enter BMI: "))
children = float(input("Enter number of children: "))
smoker = float(input("Enter smoker (0 for no, 1 for yes): "))
region = input("Enter region (northeast, northwest, southeast, southwest): ")
if region == 'northeast':
    region = [1, 0, 0, 0]
elif region == 'northwest':
    region = [0, 1, 0, 0]
elif region == 'southeast':
    region = [0, 0, 1, 0]
else:
    region = [0, 0, 0, 1]
prediction = model.predict([[age, sex, bmi, children, smoker] + region])
if prediction[0] == 0:
    print("The person is not likely to claim insurance.")
else:
    print("The person is likely to claim insurance.")
