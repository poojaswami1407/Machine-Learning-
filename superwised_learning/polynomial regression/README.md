# Polynomial Regression(from scratch)

Polynomial Regression is an extension of Linear Regression used to model non-linear relationships.

Instead of fitting a straight line, we fit a curve to the data.


Model:

We extend the linear model by adding polynomial features:

f(x) = w₁x + w₂x² + w₃x³ + ... + wₙxⁿ + b  


Dataset:

Data is loaded from a CSV file  
Contains:
  x_train (input feature)  
  y_train (target value)  

The relationship is non-linear, so linear regression alone is not sufficient.

Feature Engineering:

We create new features from the original feature:

x → x², x³, x⁴, x⁵, x⁶, x⁷  

Then combine all features into a matrix:

X = [x, x², x³, x⁴, x⁵, x⁶, x⁷]

Normalization:

Since polynomial features can grow very large, we normalize data:

X = (X - mean) / standard deviation  

This helps:
Faster convergence  
Stable gradient descent  

Cost Function (MSE):

J = (1 / 2m) Σ (y_pred - y)²  

Gradient Descent:

We update parameters using:

dW = (1/m) Xᵀ (y_pred - y)  
db = (1/m) Σ (y_pred - y)  

Update:

w = w - α * dW  
b = b - α * db  

Working Steps:

1. Load dataset from CSV  
2. Create polynomial features  
3. Normalize features  
4. Initialize weights and bias  
5. Compute predictions  
6. Calculate cost (MSE)  
7. Compute gradients  
8. Update parameters  
9. Repeat for multiple iterations  

Visualization:

Scatter plot of original data  
Curve fitting using predicted values  
Cost vs iterations graph  



Key Points:

Captures non-linear relationships  
Uses feature engineering  
Same cost function as linear regression  
Requires normalization for better performance  
Degree of polynomial controls model complexity  

Output:
Learned weights (w)  
Bias (b)  
Fitted curve on data  
Decreasing cost over iterations  

Note:

Higher degree = more complex model  
Too high degree = overfitting  
Proper balance is important  

