{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "     sepal_length  sepal_width  petal_length  petal_width    species\n",
      "0             5.1          3.5           1.4          0.2     setosa\n",
      "1             4.9          3.0           1.4          0.2     setosa\n",
      "2             4.7          3.2           1.3          0.2     setosa\n",
      "3             4.6          3.1           1.5          0.2     setosa\n",
      "4             5.0          3.6           1.4          0.2     setosa\n",
      "5             5.4          3.9           1.7          0.4     setosa\n",
      "6             4.6          3.4           1.4          0.3     setosa\n",
      "7             5.0          3.4           1.5          0.2     setosa\n",
      "8             4.4          2.9           1.4          0.2     setosa\n",
      "9             4.9          3.1           1.5          0.1     setosa\n",
      "10            5.4          3.7           1.5          0.2     setosa\n",
      "11            4.8          3.4           1.6          0.2     setosa\n",
      "12            4.8          3.0           1.4          0.1     setosa\n",
      "13            4.3          3.0           1.1          0.1     setosa\n",
      "14            5.8          4.0           1.2          0.2     setosa\n",
      "15            5.7          4.4           1.5          0.4     setosa\n",
      "16            5.4          3.9           1.3          0.4     setosa\n",
      "17            5.1          3.5           1.4          0.3     setosa\n",
      "18            5.7          3.8           1.7          0.3     setosa\n",
      "19            5.1          3.8           1.5          0.3     setosa\n",
      "20            5.4          3.4           1.7          0.2     setosa\n",
      "21            5.1          3.7           1.5          0.4     setosa\n",
      "22            4.6          3.6           1.0          0.2     setosa\n",
      "23            5.1          3.3           1.7          0.5     setosa\n",
      "24            4.8          3.4           1.9          0.2     setosa\n",
      "25            5.0          3.0           1.6          0.2     setosa\n",
      "26            5.0          3.4           1.6          0.4     setosa\n",
      "27            5.2          3.5           1.5          0.2     setosa\n",
      "28            5.2          3.4           1.4          0.2     setosa\n",
      "29            4.7          3.2           1.6          0.2     setosa\n",
      "..            ...          ...           ...          ...        ...\n",
      "120           6.9          3.2           5.7          2.3  virginica\n",
      "121           5.6          2.8           4.9          2.0  virginica\n",
      "122           7.7          2.8           6.7          2.0  virginica\n",
      "123           6.3          2.7           4.9          1.8  virginica\n",
      "124           6.7          3.3           5.7          2.1  virginica\n",
      "125           7.2          3.2           6.0          1.8  virginica\n",
      "126           6.2          2.8           4.8          1.8  virginica\n",
      "127           6.1          3.0           4.9          1.8  virginica\n",
      "128           6.4          2.8           5.6          2.1  virginica\n",
      "129           7.2          3.0           5.8          1.6  virginica\n",
      "130           7.4          2.8           6.1          1.9  virginica\n",
      "131           7.9          3.8           6.4          2.0  virginica\n",
      "132           6.4          2.8           5.6          2.2  virginica\n",
      "133           6.3          2.8           5.1          1.5  virginica\n",
      "134           6.1          2.6           5.6          1.4  virginica\n",
      "135           7.7          3.0           6.1          2.3  virginica\n",
      "136           6.3          3.4           5.6          2.4  virginica\n",
      "137           6.4          3.1           5.5          1.8  virginica\n",
      "138           6.0          3.0           4.8          1.8  virginica\n",
      "139           6.9          3.1           5.4          2.1  virginica\n",
      "140           6.7          3.1           5.6          2.4  virginica\n",
      "141           6.9          3.1           5.1          2.3  virginica\n",
      "142           5.8          2.7           5.1          1.9  virginica\n",
      "143           6.8          3.2           5.9          2.3  virginica\n",
      "144           6.7          3.3           5.7          2.5  virginica\n",
      "145           6.7          3.0           5.2          2.3  virginica\n",
      "146           6.3          2.5           5.0          1.9  virginica\n",
      "147           6.5          3.0           5.2          2.0  virginica\n",
      "148           6.2          3.4           5.4          2.3  virginica\n",
      "149           5.9          3.0           5.1          1.8  virginica\n",
      "\n",
      "[150 rows x 5 columns]\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np \n",
    "\n",
    "df = pd.read_csv('dataset.csv')\n",
    "print(df)"
   ]
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
