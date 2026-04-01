# Neural Network from Scratch (NumPy)

This project contains two implementations of Neural Networks built from scratch using NumPy for binary classification tasks.  
The goal is to deeply understand how neural networks work internally — without using frameworks like TensorFlow or PyTorch.


# 1: Dummy Classification Neural Network

Overview:
In this project, a synthetic dataset is created using `make_classification` from sklearn.  
The dataset includes:
Low class separation
Noise (label flipping)

This makes the problem more realistic and slightly difficult, helping us understand how neural networks handle imperfect data.


Model Architecture:

Input Layer: 2 features  
Hidden Layer 1: 16 neurons (ReLU)  
Hidden Layer 2: 16 neurons (ReLU)  
Output Layer: 1 neuron (Sigmoid)



Key Steps Explained:
1. Data Generation:
`make_classification()` is used to create a binary dataset
Noise is added using `flip_y=0.1`
This simulates real-world imperfect data

2. Data Preprocessing:
Features are standardized:
  
  X = (X - mean) / std
  
Helps in faster and stable training


3. Forward Propagation
Each layer performs:

Z = XW + b  
A = activation(Z)

ReLU is used in hidden layers → handles non-linearity
Sigmoid is used in output → gives probability (0–1)


4. Loss Function
Binary Cross Entropy:

Loss = -[y log(ŷ) + (1-y) log(1-ŷ)]

Measures how far predictions are from actual values

5. Backpropagation
Gradients are calculated manually using chain rule
Errors are propagated backward layer by layer
Updates:
  
  W = W - α * dW  
  b = b - α * db  

6. Training
Gradient Descent is used
Model runs for multiple iterations
Loss decreases over time

7. Decision Boundary
Meshgrid is created
Model predicts on each point
Boundary shows how model separates classes

Output:
Dataset visualization
Cost vs Iterations graph
Decision Boundary plot

What You Learn:
Complete working of Neural Networks
Handling noisy data
Multi-layer learning
Decision boundary formation

# 2: Coffee Roasting Classification Neural Network

Overview:
This project classifies whether coffee is **properly roasted or not** based on:
Temperature
Roasting Duration

Dataset is manually created using logical rules → making it a **non-linear classification problem**

Model Architecture:

Input Layer: 2 features  
Hidden Layer: 3 neurons (ReLU)  
Output Layer: 1 neuron (Sigmoid)

Key Steps Explained:

1. Custom Dataset Creation
Random values generated
Conditions applied:
  Temperature range: 175–260°C
  Time: 12–15 minutes
Labels assigned based on rules

This creates a **non-linear pattern**

2. Feature Scaling
Same standardization applied:

X = (X - mean) / std


3. Weight Initialization
He Initialization used:

sqrt(2 / (input + output))

Helps in better convergence compared to random small values

4. Forward Propagation
Same process:
Linear → Activation
ReLU + Sigmoid



5. Backpropagation
Gradients computed manually
Hidden layer learns internal patterns
6. Training
Higher learning rate used (0.5)
More iterations (10000)
Model learns complex boundaries

7. Decision Boundary (Important ):
Instead of final output, hidden neurons (z1, z2, z3) are plotted:

Each neuron creates its own boundary
Combined → final complex classification

This shows how hidden layers transform data


Output:
Coffee dataset plot
Cost vs Iterations graph
Multiple decision boundaries (z1, z2, z3)


What You Learn:
Non-linear classification
Role of hidden neurons
Feature transformation
Importance of initialization


Technologies Used:

Python  
NumPy  
Matplotlib  
Scikit-learn (only for Project 1)

