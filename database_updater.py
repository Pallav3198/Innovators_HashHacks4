

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow

from pymongo import MongoClient 
  
client=MongoClient('localhost',27017)
db = client.testdatabase
courses = db.courses

import warnings
warnings.filterwarnings('ignore')

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle 
dataset = pd.read_csv("heart.csv") 

dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
    dt_classifier.fit(X_train, y_train)
    dt_scores.append(dt_classifier.score(X_test, y_test))

with open('model_pickle','wb') as f:
    heart_model = pickle.dumps(dt_classifier)











emp_rec2 = { 
        "age":18, 
        "sex":"Male",
         "ID":"888267282",
        'Prescription':"Antibiotics,Paracetamol",
        'Symptoms':"headache ,fever",
        "Diagnosis":"Common flu",
        "bloodtest": [42,0,2,120,209,0,1,173,0,0,1,0,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ] ,
        "prediction": ""
        } 
  
emp_rec3 = { 
        "age":18, 
        "sex":"Male",
         "ID":"88826726682",
        'Prescription':"Antibiotics,Paracetamol",
        'Symptoms':"headache ,fever",
        "Diagnosis":"Common flu",
        "bloodtest": [42,0,2,120,209,0,1,173,0,0,1,0,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ] ,
        "prediction": ""
        } 
# Insert Data 
courses.insert_one(emp_rec2) 
courses.insert_one(emp_rec3) 





course = courses.find() 
for record in course: 
    x=record
    #print(x["ID"])
    
    


    if(dt_classifier.predict([x["bloodtest"]])):
        print('ok')
        courses.update_one({ "ID":x["ID"]},{
                      '$set': {
                        'prediction': "Detected"
                              }
                            }, upsert=True)
    else:
        print("no")
        courses.update_one({"ID":x["ID"]},{
                      '$set': {
                        'prediction': "not Detected"
                              }
                            }, upsert=True)
        
