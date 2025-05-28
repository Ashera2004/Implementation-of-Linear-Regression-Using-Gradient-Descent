# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.


## Program:
```
/*
Program to implement the linear regression using gradient descent.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Linear regression function
def linear_regression(x1, y, learning_rate=0.01, num_iters=1000):
    x = np.c_[np.ones(len(x1)), x1]  
    theta = np.zeros((x.shape[1], 1)) 
    for _ in range(num_iters):
        predictions = x.dot(theta)
        errors = predictions - y
        theta -= (learning_rate / len(x1)) * x.T.dot(errors)
    return theta

# Load data
data = pd.read_csv('50_Startups.csv', header=0)  
print(data.head())

X= (data.iloc[1:, :-2].values)
print("Values of Independent Variables(X):")
print()
print(X)

X1=X.astype(float)
scaler_X = StandardScaler()
Y = (data.iloc[1:,-1].values).reshape(-1,1)
print("Values of Dependent Variables(Y):")
print()
print(Y)

X1_Scaled = scaler_X.fit_transform(X1)
scaler_Y = StandardScaler()
Y1_Scaled = scaler_Y.fit_transform(Y)
print("X1_Scaled:")
print()
print(X1_Scaled)
print()
print("Y1_Scaled:")
print()
print(Y1_Scaled)

theta = linear_regression(X1_Scaled, Y1_Scaled)

#predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(1,-1)
new_Scaled = scaler_X.transform(new_data)
predictions = np.c_[np.ones(1), new_Scaled].dot(theta)
pre = scaler_Y.inverse_transform(predictions)
print(f"Prediction value:{pre}")

Developed by: A S Siddarth
RegisterNumber: 212224040316 
*/
```

## Output:

Data Information

![exp3_out1](https://github.com/user-attachments/assets/048fbbbf-bb15-4933-8b14-8c3692389b53)

![exp3_out2](https://github.com/user-attachments/assets/04d81fdd-985f-4ef9-8414-86f0062713c8)

![exp3_out3](https://github.com/user-attachments/assets/a3368cc4-1fd2-4bb6-93a2-1225192e4012)

![exp3_out4](https://github.com/user-attachments/assets/9ff69a49-917a-4bd1-8bf4-582d69929620)

![exp3_out5](https://github.com/user-attachments/assets/1a9db2bc-db5e-490b-aa7e-ff3ae4abd5fc)

![exp3_out6](https://github.com/user-attachments/assets/a5facab9-5b90-4acd-88ee-2330b339e473)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
