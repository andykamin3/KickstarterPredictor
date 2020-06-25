### Joaquin Carriquiry Castro y Andrés Kaminker
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv("ks_data.csv")
df.describe()