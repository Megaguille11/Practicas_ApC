from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
import seaborn as sns


# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset


def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t


# Carreguem dataset d'exemple
dataset = load_dataset('Life Expectancy Data.csv')
data = dataset.values

x = data[:, :2]
y = data[:, 2]

print("Dimensionalitat de la BBDD:", dataset.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)

# Evolució de l'esperança de vida global
plt.figure()

ax = plt.scatter(x[:,1], y)

# Evolució en cada país en concret
plt.figure(figsize=(55,5))
plt.xticks(rotation=90)

ax = plt.scatter(x[:,0], y)

# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
correlacio = dataset.corr()

plt.figure(figsize=(13,13))

ax = sns.heatmap(correlacio, annot=True, linewidths=.5)

# Estandaritzem les dades
x_t = standarize(x)
