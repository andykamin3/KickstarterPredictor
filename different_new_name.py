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
new_model = tf.keras.models.load_model('saved_model/my_model')


# Check its architecture
new_model.summary()