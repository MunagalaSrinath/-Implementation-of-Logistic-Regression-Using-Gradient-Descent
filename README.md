# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the data file and import numpy, matplotlib and scipy.
2. Visulaize the data and define the sigmoid function, cost function and gradient descent.
3. Plot the decision boundary .
4. Calculate the y-prediction.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SARAVANAKUMAR
RegisterNumber: 212222230133 
*/

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
x=data1.iloc[:, : -1]
y=data1["status"]
theta=np.random.randn(x.shape[1])
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,x,y):
  h=sigmoid(x.dot(theta))
  return -np.sum(y*np.log(h)+(1-y)*np.log(1-h)) 
def gradient_descent(theta,x,y,alpha,num_iterations):
  m=len(y)
  for i in range (num_iterations):
    h=sigmoid(x.dot(theta))
    gradient=x.T.dot(h-y)/m
    theta-=alpha*gradient
  return theta
theta=gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000)
def predict(theta,x):
  h=sigmoid(x.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred
y_pred=predict(theta,x)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y.values)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print("Prdicted Result:",y_prednew)
```

## Output:
![image](https://github.com/Saravana-kumar369/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/117925254/20d4f9c8-6b5a-4bda-b13a-a4eaf2903d2a)

![image](https://github.com/Saravana-kumar369/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/117925254/5d17e129-9dbf-4f26-843a-e05ba9803da1)

![image](https://github.com/Saravana-kumar369/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/117925254/0b8af54c-68ae-4b00-883a-6d5a17846012)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

