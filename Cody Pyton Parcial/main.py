import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pylab
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.stats as sci
from scipy.stats import stats
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import r2_score
dataset = pd.read_csv('Life Expectancy Data.csv', header=0, delimiter=',')
cols = dataset.columns
def netejarDades(dataset):
    dataset2 = dataset.drop(['Country'], axis=1)
    dataset3 = dataset2.drop(['Status'], axis=1)
    dataset3 = dataset3.astype('float64')
    for i in dataset3.columns:
        c=dataset3[i].mean()
        dataset3[i] = dataset3[i].fillna(c)

    return dataset3

# plots the distribution of all the attributes

def mse(v1, v2):
    return ((v1 - v2)**2).mean()


def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t

def GaussianDistribution_Plot2(x, cols):
    for i in range(x.shape[1]):
        if np.unique(x[:, i]).shape[0] > 2:
            plt.figure()
            plt.title("")
            plt.xlabel("Attribute Value: " + cols[i])
            plt.ylabel("Count")
            hist = plt.hist(x[:, i], bins=15, range=[np.min(x[:, i]), np.max(x[:, i])], histtype="bar", rwidth=0.8)
            plt.show()




def GaussianDistribution(x, cols):
    a = np.array([])
    for i in range(x.shape[1]):
        p = stats.skew(x[:, i])
        print("Atribut: " + cols[i] + " Valor de χ2: ")
        print(p)
        if p < 0.05:
            a = np.append(a, [cols[i]], axis=0)

    return a

def regression(x, y):
    # Creem un objecte de regressió de sklearn
    regr = LinearRegression()

    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)

    # Retornem el model entrenat
    return regr

def regresor_lineal(x, y):
    atribut1 = x[:, 0].reshape(x.shape[0], 1)
    x = np.delete(x, 1, axis=1)
    regr = regression(x, atribut1)
    predicted = regr.predict(x)

    # Mostrem la predicció del model entrenat en color vermell a la Figura anterior 1
    plt.figure()
    ax = plt.scatter(x[:, 0], y)
    plt.plot(atribut1[:, 0], predicted, 'r')
    plt.show()
    # Mostrem l'error (MSE i R2)
    MSE = mse(y, predicted)
    r2 = r2_score(y, predicted)

    print("Mean squeared error: ", MSE)
    print("R2 score: ", r2)
def main():
    dataset44 = netejarDades(dataset)
    dataset44, t=GaussianDistribution(dataset44.values, dataset44.columns)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(dataset44)
    data_scaled = standarize(data_scaled)
    #print(GaussianDistribution(data_scaled, cols))

    data_df = pd.DataFrame(data_scaled)
    #print(data_df.agg(['std', 'mean']))
    x = data_scaled[:, :]
    y = data_scaled[:, 1]
    x = np.delete(x, 1, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.45, random_state=42)
    regresor_lineal(x, y)

    return 0

main()