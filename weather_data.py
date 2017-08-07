import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

labelled = pd.read_csv("monthly-data-labelled.csv")
unlabelled = pd.read_csv("monthly-data-unlabelled.csv")

X = labelled.drop('city',1).values
y = labelled['city'].values

X_train, X_test, y_train, y_test = train_test_split(X, y)
