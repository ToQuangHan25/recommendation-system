import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self, lr=0.1, epochs=100):
        self.lr=lr
        self.epochs=epochs
        self.w=None
        self.b=None
    def loss_function(self, y_true, y_pred):
        return np.mean((y_pred-y_true)**2)
    def gradient(self, X, y):
        samples=X.shape[0]
        y_pred=self.predict(X)
        db=2/samples*np.sum(y_pred-y)
        dw=2/samples*X.T.dot(y_pred-y)
        return db, dw
    def fit(self, X, y):
        losses=[]
        self.b, self.w=0.0, np.zeros(X.shape[1])
        for epoch in range(1,self.epochs+1):
            y_pred=self.predict(X)
            db, dw=self.gradient(X, y)
            self.b-=self.lr*db
            self.w-=self.lr*dw
            loss=self.loss_function(y, y_pred)
            losses.append(loss)
            # if epoch%100==0:
            #     print(f'epoch {epoch}, loss: {loss}')
        return self.b, self.w, losses
    def predict(self, X):
        y_pred=X.dot(self.w)+self.b
        return y_pred
    def plot_learning_curve(self, losses):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(losses)), losses, label='training loss (MSE)')
        plt.xlabel('epochs')
        plt.ylabel('MSE loss')
        plt.title('Learning curve')
        plt.legend()
        plt.grid()
        plt.show()
    def evaluate(self, y_true, y_pred):
        MAE=np.mean(np.abs(y_pred-y_true))
        MSE=np.mean((y_pred-y_true)**2)
        RMSE=np.sqrt(MSE)
        R2=1-np.sum((y_pred-y_true)**2)/np.sum((np.mean(y_true)-y_true)**2)
        return MAE, MSE, RMSE, R2