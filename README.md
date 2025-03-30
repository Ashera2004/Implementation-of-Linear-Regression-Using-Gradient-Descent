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
Developed by: 
RegisterNumber:  
*/
```

## Output:
![linear regression using gradient descent](sam.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
