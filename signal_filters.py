import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

class RollngMeanFilter(BaseEstimator, TransformerMixin):
    def __init__(self, width=5, drop=False):
        self.width = width
        self.drop = drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y = None):
        result = pd.DataFrame()
        for col in X.columns:
            data = np.array(X[col])
            if self.drop == False:
                data = np.concatenate([np.zeros(self.width-1), data])
                result[col+'_roll_mean_'+str(self.width)] = X[col].rolling(self.width).mean()
            else:
                result[col+'_roll_mean_'+str(self.width)] = X[col].rolling(self.width, min_periods=self.width).mean().dropna()                
        return result
    
class MedianFilter(BaseEstimator, TransformerMixin):
    def __init__(self, width=5, drop=False):
        self.width = width
        self.drop = drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y = None):
        result = pd.DataFrame()
        for col in X.columns:
            data = np.array(X[col])
            if self.drop == False:
                data = np.concatenate([np.zeros(self.width-1), data])
                result[col+'_median_'+str(self.width)] = X[col].rolling(self.width).median()
            else:
                result[col+'_median_'+str(self.width)] = X[col].rolling(self.width, min_periods=self.width).median().dropna()                
        return result
    
class ExponentialFilter(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y = None):
        result = pd.DataFrame()
        for col in X.columns:
            data = np.array(X[col])
            col_result = [data[0]]
            for n in range(1, len(data)):
                col_result.append(self.alpha * data[n] + (1 - self.alpha) * col_result[n-1])
            result[col+'_exp'] = col_result
        return result
    
class DoubleExponentialFilter(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.05, beta=0.5):
        self.alpha = alpha
        self.beta = beta
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y = None):
        result = pd.DataFrame()
        for col in X.columns:
            data = np.array(X[col])
            col_result = [data[0]]
            for n in range(1, len(data)+1):
                if n == 1:
                    level, trend = data[0], data[1] - data[0]
                if n >= len(data): # прогнозируем
                    value = col_result[-1]
                else:
                    value = data[n]
                last_level, level = level, self.alpha*value + (1-self.alpha)*(level+trend)
                trend = self.beta*(level-last_level) + (1-self.beta)*trend
                col_result.append(level+trend)
            result[col+'_exp2'] = col_result
        return result


def generate_true_data(length=30, resolution=20):
    sinus_g = [np.sin(i / resolution) for i in range(length * resolution)]

    square_g = [(1 if p > 0 else -1) for p in sinus_g]

    triangle_g = []
    t = -1
    for _ in range(length * resolution):
        t = t+0.035 if t < 1 else -1
        triangle_g.append(t)

    data = pd.DataFrame({'sinus':sinus_g, 'square':square_g, 'triangle':triangle_g})
    return data

def add_noise(data, k=0.3, fitob=0.03):
    noised_data = pd.DataFrame()
    for col in data.columns:
        r = (np.random.random(size=len(data[col]))*2-1) * k

        # Standard noise and random emissions
        emissions_proba = np.random.random(size=len(data[col]))
        koef = np.array([7 if x < fitob else 1 for x in emissions_proba])
        r *= koef
        noised_data[col+'_noised'] = data[col] + r
    return noised_data
    

if __name__ == '__main__':
    length=30
    resolution=20
    data = generate_true_data(length=30, resolution=20)
    noised_data = add_noise(data)
    
    #filter = RollngMeanFilter(width=5, drop=False)
    #filter = MedianFilter(width=5, drop=False)
    #filter = ExponentialFilter(alpha=0.1)
    filter = DoubleExponentialFilter(alpha=0.1, beta=0.1)
    filtered_data = filter.fit_transform(data)
    
    fig, axs = plt.subplots(3, 1)
    for i, col in enumerate(data.columns):
        axs[i].plot(data[col])
        axs[i].plot(noised_data[col+'_noised'])
        axs[i].plot(filtered_data[filtered_data.columns[i]])
        axs[i].set_ylim([-2, 2]), axs[i].set_xlim([0, length*resolution]),
        axs[i].set_yticklabels([]), axs[i].set_xticklabels([])
    plt.show() 