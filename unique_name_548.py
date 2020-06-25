### Joaquin Carriquiry Castro y Andrés Kaminker
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
#Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree
#Knn
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib as plt
df = pd.read_csv("ks_data.csv")


category_list = df['category'].drop_duplicates()
#cat_counts = df['category'].value_counts()
value_map = {}
x=0
for category in category_list:
  value_map[category] = x
  x += 1

campaign_category = df['main_category'].drop_duplicates()
#cam_counts = df['category'].value_counts()
cam_map = {}
x=0
for cat in campaign_category:
    cam_map[cat] = x
    x += 1


country_list = df['country'].drop_duplicates()
#cam_counts = df['category'].value_counts()
country_map = {}
x=0
for country in country_list:
    country_map[country] = x
    x += 1

country_list = df['country'].drop_duplicates()
#cam_counts = df['category'].value_counts()
country_map = {}
x=0
for country in country_list:
    country_map[country] = x
    x += 1


currency_list = df['currency'].drop_duplicates()
#cam_counts = df['category'].value_counts()
currency_map = {}
x=0
for currency in currency_list:
    currency_map[currency] = x
    x += 1

df.replace({"currency": currency_map, 'country': country_map, 'main_category': cam_map, 'category': value_map}, inplace=True )
df.astype({"currency": 'int32', 'country': 'int32', 'main_category': 'int32', 'category': 'int32'})
df.head()

df['launched'] = pd.to_datetime(df['launched'], errors='raise')
df['deadline'] = pd.to_datetime(df['deadline'], errors='raise')
df['deltatime'] = (df['deadline'] - df['launched'])
df['deltatime'] = df['deltatime'].dt.days


df.replace(to_replace = "successful", inplace = True, value=1 )
df.replace(to_replace = "canceled", inplace = True, value=0)
df.replace(to_replace = "failed", inplace = True, value=0)
df["state"] = pd.to_numeric(df["state"], errors='coerce')


df.drop(["goal","ID", "name","deadline","launched","country", "pledged", "backers", "usd pledged", "usd_pledged_real"], axis=1, inplace=True) #We also have to dropped pledged and backers in order to train the model for a common starting point (t=0)
df.dropna(inplace = True)
#print(df.describe())
ty = df["state"]
features = list(df.columns.values)
features.remove("state")
tx = df[features]
scaler = StandardScaler()
print(df.dtypes)
X = np.asarray(tx)
y = np.asarray(ty)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(5,)),
    keras.layers.Dense(2, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=10)
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

model.save('saved_model/my_model') 
'''
print("Tree: ")
clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train, y_train)
#tree.plot_tree(clf)
y_pred = clf.predict(X_test)

plot_tree(clf)
print("Acurracy: ", metrics.accuracy_score(y_test, y_pred))
print("Precision", metrics.precision_score(y_test, y_pred));
print("Recall: ", metrics.recall_score(y_test, y_pred))
print( metrics.classification_report(y_test, y_pred))
print( metrics.confusion_matrix(y_test, y_pred))
print("F1", metrics.f1_score(y_test, y_pred));
'''
