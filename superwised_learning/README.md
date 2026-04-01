## Supervised Learning

This folder contains implementations of fundamental Supervised Learning algorithms.  
Supervised learning is a type of machine learning where the model is trained on labeled data.

Subfolders

1. Linear Regression  
2. Logistic Regression  
3. Polynomial Regression  
4. Polynomial Logistic Regression  
5. Neural Network  
6. Decision Tree  

# 1. Linear Regression

Linear Regression is used for predicting continuous values.

We try to fit a straight line to the data using:

f(x) = w * x + b  

Where:  
w = weight (slope)  
b = bias (intercept)  

Cost Function:
We use Mean Squared Error (MSE):

J(w, b) = (1/m) Σ (y_pred - y_actual)²  

Optimization:
We use Gradient Descent to update w and b:

w = w - α * ∂J/∂w  
b = b - α * ∂J/∂b  

Output:
Continuous values  
Range: (-∞, +∞)

# 2. Logistic Regression

Logistic Regression is used for binary classification (0 or 1).

Model:

z = w * x + b  

Apply sigmoid function:

g(z) = 1 / (1 + e^(-z))  

Output:
Value between 0 and 1 (probability)  
Final prediction:
  ≥ 0.5 → 1  
  < 0.5 → 0  

Cost Function (Log Loss):
J = - [ y log(y_pred) + (1 - y) log(1 - y_pred) ]


# 3. Polynomial Regression

Polynomial Regression is an extension of Linear Regression.

We create new features from existing ones:

Example:  
x → x², x³, x⁴ ...

Model still remains linear in terms of weights.

Key Points:
Feature engineering is important  
Can capture non-linear relationships  
Uses same:
  Cost Function (MSE)  
  Gradient Descent  

# 4. Polynomial Logistic Regression

This is Logistic Regression with polynomial features.

First create polynomial features  
Then apply logistic regression  

Example:
x → x², x³  

Why use it?
To handle non-linear decision boundaries  

Output:
Still classification (0 or 1)  
Uses sigmoid + log loss  

# 5. Neural Network

Neural Networks are inspired by the human brain and consist of layers:

Input Layer  
Hidden Layers  
Output Layer  

Working:
Each neuron performs:

z = w * x + b  
a = activation(z)  

Activation Functions:
Sigmoid  
ReLU  
Tanh  

Training:
Forward Propagation  
Loss Calculation  
Backpropagation  
Gradient Descent  

Use Cases:
Image recognition  
NLP  
Complex pattern learning  

# 6. Decision Tree

Decision Tree is a tree-based model used for both classification and regression.

Structure:
Root Node  
Decision Nodes  
Leaf Nodes  

Working:
Splits data based on features  Uses criteria like:
  Gini Index  
  Entropy (Information Gain)  

Advantages:
Easy to understand  
No need for feature scaling  

Disadvantages:
Can overfit easily  
