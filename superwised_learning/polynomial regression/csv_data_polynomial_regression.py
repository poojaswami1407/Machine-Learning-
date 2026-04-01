import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_csv(r"C:\Users\Hp\OneDrive\Desktop\python\machine_learning_course1\polynomial_data.csv")
print(data)
x = data["x_train"].values
x_col = x.reshape(-1, 1)
y = data["y_train"].values
y_col = y.reshape(-1, 1)
plt.scatter(x_col,y_col)
plt.show()

print("x_col shape : ", x_col.shape)
print("y_col shape : ", y_col.shape)

x_2 = x_col**2
print("x_2 shape : ", x_2.shape)

x_3 = x_col**3
print("shape of x_3 : ", x_3.shape)

x_4 = x_col**4
print("shape of x_4 : ", x_4.shape)

x_5 = x_col**5
print("shape of x_5 : ", x_5.shape)

x_6 = x_col**6
print("shape of x_ : ", x_6.shape)

x_7 = x_col**7
print("shape of x_7 : ", x_7.shape)

# creating weigths and bias
w = np.array([[0], [0], [0], [0], [0], [0], [0]])
print("the shape of w : ",w.shape)

b = 0 
m = len(y_col)
alpha = 0.02
x_matrix = np.hstack((x_col, x_2, x_3, x_4, x_5, x_6, x_7))

# normalization
x_mean = np.mean(x_matrix, axis = 0)
x_sd = np.std(x_matrix, axis = 0)
x_matrix = (x_matrix - x_mean) / x_sd

# compute prediction
def compute_prediction(w, x_matrix, b):
    y_pred = (x_matrix @ w + b)
    return y_pred

#compute cost 
def compute_cost(y_col,y_pred):
    c = y_pred - y_col
    cost = np.sum(c**2)
    final_cost = (1/(2*m)) * cost
    return final_cost, c



# gradient descent 
def gradient_decent(x_matrix, c):
    d_w = (x_matrix.T @ c) / m
    d_b = np.sum(c) / m
    return d_w, d_b

costs = []

for i in range(50000):
    y_pred = compute_prediction(w, x_matrix,b)
    final_cost, c = compute_cost(y_col,y_pred)
    costs.append(final_cost)
    d_w, d_b = gradient_decent(x_matrix, c)
    w = w - alpha * d_w
    b = b - alpha * d_b
    #print(final_cost)

plt.plot(costs)
plt.show()
plt.scatter(x_col,y_col)
plt.plot(x_col.ravel(), y_pred.ravel())
plt.show()