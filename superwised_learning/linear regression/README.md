# Linear Regression Implementation (From Scratch)

This project demonstrates the implementation of Linear Regression using Gradient Descent from scratch using NumPy.

It includes:
1. Single Variable Linear Regression
2. Multiple Variable Linear Regression


# 1. Single Variable Linear Regression

Overview:
In this model, we predict the target using only one feature.

Mathematical model:

f(x) = w * x + b  

Where:
w = slope (weight)
b = intercept (bias)

Dataset Generation:
Random values of x are generated
Target is created using:

y = 4 + 3x + noise  

This simulates real-world noisy data.

Implementation Steps:

1. Prediction Function:
We compute predicted values using:

y_pred = w * x + b  

2. Cost Function (MSE):

J = (1 / 2m) Σ (y_pred - y)²  

Measures how far predictions are from actual values  
Goal: minimize this cost  

3. Gradient Descent:

We update parameters using:

w = w - α * dj_dw  
b = b - α * dj_db  

Where:
α → learning rate  
learning rate is that how much we need to change our weights and bias 

Key Points:
Uses loop-based implementation (beginner-friendly)
Helps understand internal working of ML models
Output is continuous (regression)


# 2. Multiple Variable Linear Regression

Overview:
In this model, we use multiple features to predict the output.

Mathematical model:

f(x) = w₁x₁ + w₂x₂ + w₃x₃ + ... + b  

Vector form:

y = XW + b  

Dataset Generation:
3 input features are generated
True weights:

[3, 2, 1]

Target:

y = 4 + (3x₁ + 2x₂ + 1x₃) + noise  

Implementation Steps:

1. Prediction (Vectorized):

y_pred = X @ W + b  

Uses matrix multiplication (efficient)

2. Cost Function:

J = (1 / 2m) Σ (y_pred - y)²  

Same as single variable but vectorized

3. Gradient Descent:

dW = (1/m) Xᵀ (y_pred - y)  
db = (1/m) Σ (y_pred - y)  

Update:

W = W - α * dW  
b = b - α * db  

Key Points:
Uses vectorized operations (fast & optimized)
Scales to higher dimensions easily
Real-world ML models use this approach

Visualization:

Scatter plot used to visualize training data  
Cost vs Iteration graph shows learning progress  
In multiple regression, visualization is done using only one feature (for simplicity)

Learning Outcomes:

By completing this project, you will understand:

How Linear Regression works internally  
Difference between single and multiple variables  
Gradient Descent optimization  
Importance of vectorization in ML  


Note:

Learning rate (α) is important for convergence  
Too high → divergence  
Too low → slow learning  


