import pandas as pd
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load the fixed CSV
train_df = pd.read_csv("machine_learnng_course_2/mnist_train.csv")
test_df = pd.read_csv("machine_learnng_course_2/mnist_test.csv")

# One example per class
examples = train_df.groupby("label").first().reset_index()
#
y = train_df.iloc[:,0]
X = train_df.iloc[:,1:]  
y = pd.get_dummies(y)

x_test = test_df.iloc[:,1:] 
y_test = test_df.iloc[:,0]
y_test = pd.get_dummies(y_test)

#converting into numpy arrys 
X = X.values
y = y.values

x_test = x_test.values
y_test = y_test.values

#normalization
X = X / 255.0
x_test = x_test / 255.0

# Plot
plt.figure(figsize=(10, 4))
for i in range(10):
    ax = plt.subplot(2, 5, i + 1)
    img = examples.loc[i].drop("label").values.astype(np.uint8).reshape(28, 28)
    plt.imshow(img, cmap="gray")
    plt.title(f"Label: {examples.loc[i, 'label']}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# activation function 
def relu(z):
    a = np.maximum(0,z)
    return a

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

#derivative of activation functions 
def derivative_relu(z):
    return (z > 0).astype(float)


#creating layers
layers = [512,256,10]
activation_functions = [relu, relu, softmax]
derivative_activation = [derivative_relu,derivative_relu]

#creating weights and bias
w = []
b = []

x_1 = X.shape[1]

for i in range(len(layers)):       # for weights
    if i == 0:
        a = np.random.randn(x_1, layers[i]) * np.sqrt(2/(x_1+layers[i]))
    else:
        a = np.random.randn(layers[i-1],layers[i]) * np.sqrt(2/(layers[i-1]+layers[i]))
    w.append(a)

for i in range(len(layers)):         # for bias
    c = np.zeros((1,layers[i]))
    b.append(c)


#forward propogation
def forward_propogation(w,X,b):
    out = []
    out.append(X)
    z = []
    for i in range(len(layers)):
        k = ( out[i] @ w[i] ) + b[i]
        z.append(k)
        a_1 = activation_functions[i](k)
        out.append(a_1)
    y_p = out[-1]
    return out,z,y_p

#computing cost  
m = train_df.shape[0] 

def compute_cost(m,y_p,y):
    y = y
    y_p = y_p
    cost = -np.sum(y * np.log(y_p + 1e-8)) / m
    return cost

def gradient_descent(y_p,out,y,w,b,m,z):
    d_dw = []
    d_db = []
    for i in range(len(layers)):
        if i == 0:
            d_z = (y_p - y ) 
            d_w = out[-2].T @ d_z
            d_w = d_w / m
            d_dw.insert(0,d_w)
            d_b = np.sum(d_z, axis=0, keepdims=True)
            d_b = d_b / m
            d_db.insert(0,d_b)
        else:
            d_z = d_z @ w[-i].T * derivative_activation[-i](z[-(i+1)])
            d_w = out[-(i+2)].T @ d_z / m
            d_dw.insert(0,d_w)
            d_b = np.sum(d_z, axis=0, keepdims=True)
            d_b = d_b / m
            d_db.insert(0,d_b)
    return d_dw,d_db

alpha = 0.01
batch_size = 64
costs = []
for i in range(1000):

    perm = np.random.permutation(m)
    X = X[perm]
    y = y[perm]

    for start in range(0, m, batch_size):

        end = start + batch_size
        X_batch = X[start:end]
        y_batch = y[start:end]

        out,z,y_p = forward_propogation(w,X_batch,b)

        cost = compute_cost(batch_size,y_p,y_batch)

        d_dw,d_db = gradient_descent(y_p,out,y_batch,w,b,batch_size,z)

        for j in range(len(w)):
            w[j] = w[j] - alpha * d_dw[j]
            b[j] = b[j] - alpha * d_db[j]
            
    out_train, z_train, y_pred_train = forward_propogation(w, X, b)
    predictions_training = np.argmax(y_pred_train, axis=1)
    true_labels_training = np.argmax(y, axis=1)
    training_accuracy = np.mean(predictions_training == true_labels_training)

    out_test, z_test, y_pred_test = forward_propogation(w, x_test, b)
    test_cost = compute_cost(x_test.shape[0], y_pred_test, y_test)
 
      
    predictions_testing = np.argmax(y_pred_test, axis=1)
    true_labels_testing = np.argmax(y_test, axis=1)
    accuracy_testing = np.mean(predictions_testing == true_labels_testing)
    
    print(i,cost,training_accuracy,test_cost,accuracy_testing)




