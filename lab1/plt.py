import pandas as pd
from matplotlib import pyplot as plt

iris = pd.read_csv("dataset.csv")

irisTest = iris.sample(frac=0.3)
irisTrain = iris.sample(frac=0.7)
colors = {'setosa':'red', 'versicolor':'blue', 'virginica':'green'}
plt.scatter(irisTrain.sepal_length, irisTrain.sepal_width, c=irisTrain.species.apply(lambda x: colors[x]))
plt.xlabel("sepal length Train (cm)") 
plt.ylabel("sepal width Train(cm)")  
plt.legend(colors)
print(iris)
plt.show()
