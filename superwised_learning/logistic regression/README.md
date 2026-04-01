# Logistic Regression / Classification (from scratch)

Logistic Regression is used for binary classification problems (0 or 1).

In this implementation, we classify data into two classes using a decision boundary.


Model:
First we compute linear combination:

z = w * x + b  

Then apply sigmoid function:

g(z) = 1 / (1 + e^(-z))  

Output:
Value between 0 and 1 (probability)  
Final prediction:
  ≥ 0.5 → Class 1  
  < 0.5 → Class 0  


Dataset:

We generate dummy classification data using sklearn:

200 samples  
2 features (for visualization)  
Some noise added (real-world simulation)  


Cost Function (Log Loss):

J = - [ y log(y_pred) + (1 - y) log(1 - y_pred) ]

Note:
Small value (1e-8) is added to avoid log(0) error  

Gradient Descent:

We update weights and bias using:

dW = (1/m) Xᵀ (y_pred - y)  
db = (1/m) Σ (y_pred - y)  

Update:

w = w - α * dW  
b = b - α * db  


Working Steps:

1. Initialize weights and bias  
2. Compute prediction using sigmoid  
3. Calculate cost (log loss)  
4. Compute gradients  
5. Update parameters  
6. Repeat for multiple iterations  


Visualization:

Scatter plot:
  Red → Class 0  
  Blue → Class 1  

Cost vs Iterations graph (learning progress)

Decision Boundary:

w₁x₁ + w₂x₂ + b = 0  

Converted into:

x₂ = (-b - w₁x₁) / w₂  


Key Points:

Used for classification, not regression  
Output is probability, not direct value  
Uses sigmoid function  
Uses log loss instead of MSE  
Decision boundary separates classes  


Output:

Trained weights (w)  
Bias (b)  
Decision boundary line  
Cost decreasing over iterations  