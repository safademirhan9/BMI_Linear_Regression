import numpy as np
    
def CostFunction(x,y,w,b):
    cost_arr = (((x.dot(w) + b) - y) ** 2) / (2*len(y))
    return sum(cost_arr)

def fit(x, y, w, learning_rate, epochs):
    b = 0
    cost_list = [0] * epochs

    for epoch in range(epochs):
        z = x.dot(w) + b
        
        loss = z - y
        weight = x.T.dot(loss) / len(y)
        bias = sum(loss) / len(y)

        w = w - learning_rate * weight
        b = b - learning_rate * bias

        cost = CostFunction(x, y, w, b)
        cost_list[epoch] = cost

        if (epoch%(epochs/10)==0):
            print("Cost is:",cost)

    return w, b, cost_list

def predict(X, w, b):
    return X.dot(w) + b

def sum(arr):
    sum = 0
    for i in range(0, len(arr)):    
       sum = sum + arr[i];    
    return sum

def dot(v1, v2):
    return sum(x*y for x, y in zip(v1, v2))

def dot_product(x, y):
    dp = 0
    for i in range(len(x)):
        dp += (x[i]*y[i])
    return dp
    