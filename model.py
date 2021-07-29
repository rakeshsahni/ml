import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("flower.csv")

Lb = LabelEncoder()
df.species = Lb.fit_transform(df.species)

x_train,x_test,y_train,y_test = train_test_split(df.drop("species",axis=1),df.species,test_size=0.2)

Lg = LogisticRegression()

Lg.fit(x_train,y_train)
pred = Lg.predict(x_test)

# print(metrics.confusion_matrix(y_test,pred))

# print(Lg.predict([[5.3, 3.7, 1.5, 0.2]]))
# pickling
import pickle
with open("model.pkl","wb") as write_file :
    pickle.dump(Lg,write_file)
