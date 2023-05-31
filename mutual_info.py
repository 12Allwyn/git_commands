import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel('0-1681955587526.xlsx')

target = data.loc[:, ['Status']]

new_data = data.drop(['Status','SubStatus','State','Compliance Issues','Scenario Instance Id'], axis=1)

cols = new_data.columns

from sklearn.preprocessing import LabelEncoder

data_X = new_data.loc[: , cols]
x1 = data_X.apply(LabelEncoder().fit_transform)

y1 = target.apply(LabelEncoder().fit_transform)

from sklearn.feature_selection import mutual_info_classif

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x1, y1
    ,
    test_size=0.3,
    random_state=42)
mutual_info = mutual_info_classif(X_train, y_train.values.ravel())

mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)
# print(mutual_info.sort_values(ascending=False))

import matplotlib.pyplot as plt
mutual_info.sort_values().plot(kind='barh', color='red')
plt.xlabel('Mutual Info Score')
plt.ylabel('Feature')
plt.title('Mutual Information')
plt.show()