{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from matplotlib import pyplot as plt\n",
    "\n",
    "iris = pd.read_csv(\"dataset.csv\")\n",
    "colors = {'setosa':'red', 'versicolor':'blue', 'virginica':'green'}\n",
    "plt.scatter(iris.sepal_length, iris.sepal_width, c=iris.species.apply(lambda x: colors[x]))\n",
    "plt.xlabel(\"sepal length (cm)\") \n",
    "plt.ylabel(\"sepal width (cm)\") \n",
    "plt.legend(colors)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Hello, World!\n"
     ]
    }
   ],
   "source": [
    "print('Hello, World!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
