# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# Importing the data

gold_p = pd.read_csv('gold_price/gld_price_data.csv')

# Splitting the features and the target.
X = gold_p.drop(['Date', 'GLD'], axis=1)
y = gold_p['GLD']

# let's make train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# let's train the model now.
reg = RandomForestRegressor(n_estimators=100)

reg.fit(X_train, y_train)

# prediction
pred = reg.predict(X_test)
print(X_test)