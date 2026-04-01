# Polynomial Logistic Regression (from scratch)

Polynomial Logistic Regression is an extension of Logistic Regression used to handle non-linear classification problems.

Instead of a straight decision boundary, it creates a curved (non-linear) boundary.

Model:

First compute linear combination:

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
2 features  
Noise added (10% label flipping)  
Low class separation (hard problem)  


Feature Engineering:

We create polynomial features from original features:

x₁, x₂ → x₁², x₂²  

Final feature matrix:

X = [x₁, x₂, x₁², x₂²]

This helps model learn non-linear decision boundaries.

Cost Function (Log Loss + Regularization):

J = - [ y log(y_pred) + (1 - y) log(1 - y_pred) ]  
    + (λ / 2m) Σ w²  

Where:
λ (lambda) = regularization parameter  

Note:
Small value (1e-8) added to avoid log(0) error  

Gradient Descent:

dW = (1/m) Xᵀ (y_pred - y) + (λ/m) * w  
db = (1/m) Σ (y_pred - y)  

Update:

w = w - α * dW  
b = b - α * db  

Working Steps:

1. Generate dataset  
2. Create polynomial features  
3. Initialize weights and bias  
4. Compute predictions using sigmoid  
5. Calculate cost (log loss + regularization)  
6. Compute gradients  
7. Update parameters  
8. Repeat for multiple iterations  

Visualization:

Scatter plot:
  Class 0  
  Class 1  

Cost vs iterations graph  

Decision Boundary:
  Created using mesh grid  
  Shows curved boundary separating classes  


Decision Boundary:
Model learns boundary where:

w₁x₁ + w₂x₂ + w₃x₁² + w₄x₂² + b = 0  

This creates a non-linear separation.

Key Points:

Used for non-linear classification  
Extends logistic regression using polynomial features  
Uses regularization to prevent overfitting  
Decision boundary is curved instead of straight  
More powerful than simple logistic regression  


Output:

Learned weights (w)  
Bias (b)  
Non-linear decision boundary  
Cost decreasing over iterations  

Note:

Higher complexity → better fit but risk of overfitting  
Regularization helps control model complexity  
Feature engineering is very important  
