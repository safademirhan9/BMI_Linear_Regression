import numpy as np

class LinearReg:
    
    def __init__(self,learning_rate=0.0001,epochs=10000000):
        self.epochs = epochs
        self.lr = learning_rate
        self.w = None
        self.b = None
        self.cost_list = []
        
    def __initial_params(self,shape):
        #initialize weigth and bias as zero
        self.w = np.zeros(shape)
        self.b = 0
        return True
        
    def __predictions(self,X):
        return np.dot(X, self.w) + self.b
    
    def __calculate_cost(self,error):
        return (1/(2*error.size)) * np.dot(error.T,error)
    
    def __gradient_descent(self,X,y,y_pred):
        #difference between prediction and actual
        error = y_pred - y
        #calculate cost and append them to list
        cost = self.__calculate_cost(error)
        self.cost_list.append(cost)
        #gradients
        dw = (1 / X.shape[0]) * np.dot(X.T,error)
        db = (1 / X.shape[0]) * np.sum(error)
        return dw, db
    
    def __update_parameters(self,dw,db):
        #update weight and bias with gradients
        self.w -= self.lr * dw
        self.b -= self.lr * db
        return True
    
    def fit(self,X,y):
        self.__initial_params(X.shape[1])
        for _ in range(self.epochs):
            y_pred = self.__predictions(X)
            dw, db, = self.__gradient_descent(X, y, y_pred)
            self.__update_parameters(dw, db)
        return True
    
    def predict(self,X):
        return self.__predictions(X)
    
    def calculate_rmse(self,y_real,y_pred):
        return np.sqrt(np.mean((y_pred-y_real)**2))
        
    def calculate_r2(self,X,y):
        sum_squares = 0
        sum_residuals = 0
        y_mean = np.mean(y)
        for i in range(X.shape[0]):
            y_pred = self.__predictions(X[i])
            sum_squares += (y[i] - y_mean) ** 2
            sum_residuals += (y[i] - y_pred) ** 2
        score = 1 - (sum_residuals / sum_squares)
        return score
    
    # def dot(self, v1, v2):
    #     return sum(x*y for x, y in zip(v1, v2))
    
    # def dot_product(x, y):
    #     dp = 0
    #     for i in range(len(x)):
    #         dp += (x[i]*y[i])
    #     return dp