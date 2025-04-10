# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Initialize Parameters**:  
   - Set initial values for model parameters (θ₀, θ₁) to small random values or zeros.  
   - Define the learning rate (α) and the number of iterations for gradient descent.  

2. **Compute Cost Function**:  
   - Use the Mean Squared Error (MSE) formula to measure the difference between predicted and actual profits:
      
      ![ql_da7582f6264ec8eb575ce13e7dfe3e0e_l3](https://github.com/user-attachments/assets/16325057-ac01-4939-a280-8a8959b93e87)

   - Where    ![ql_131513049cbc3322edfe4e85f5a9e88d_l3](https://github.com/user-attachments/assets/6bac6eb5-920b-4228-8232-ca17e0bf56fd)
                           is the hypothesis function.  

3. **Perform Gradient Descent**:  
   - Update parameters iteratively using:
     
      ![ql_bcb4541e956c6b06073c7ec47bcc0a44_l3](https://github.com/user-attachments/assets/a8a84258-bfa8-4671-924f-d727cfcb38b1)

   - Repeat for the specified number of iterations or until convergence.  

4. **Train the Model**:  
   - Continue updating the parameters until the cost function stabilizes, indicating optimal values for  ![ql_3e5f55b6a59100f06e7d3685ddc0dc23_l3](https://github.com/user-attachments/assets/497b48e3-c8d8-40fb-93ed-58e5a3e4442f)


5. **Make Predictions**:  
   - Use the trained model to predict profit for a given city population by substituting values into the hypothesis function:
     
       ![ql_b782e4a6370a2cb13cbb5ff2556a7def_l3](https://github.com/user-attachments/assets/0051f671-bb0c-4ab4-82ff-5dc483db47c7)

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

#Assuming the last column is your target variable 'y' and the preceding column 
X= (data.iloc[1:, :-2].values)
print(X)

X1=X.astype(float)

scaler_X = StandardScaler()
Y = (data.iloc[1:,-1].values).reshape(-1,1)
print(Y)

X1_Scaled = scaler_X.fit_transform(X1)
scaler_Y = StandardScaler()
Y1_Scaled = scaler_Y.fit_transform(Y)

print(X1_Scaled)
print(Y1_Scaled)

#learn model parameters
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

![Screenshot 2025-03-30 174004](https://github.com/user-attachments/assets/d7638233-856c-4626-b062-e9df1b561efc)


![Screenshot 2025-03-30 174826](https://github.com/user-attachments/assets/61680617-7c0e-45e3-983b-e381811e2e11)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
