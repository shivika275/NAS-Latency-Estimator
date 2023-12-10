import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import pdb
conf_matrix = pd.read_csv('cosine_sim.csv')
conf_matrix_2 = pd.read_csv('cosine_sim.csv',header=None)
conf_matrix_2.drop(index=conf_matrix_2.index[0], inplace=True)
pdb.set_trace()
# conf_matrix_2.drop(columns=['Unnamed: 0'])
conf_matrix_2.drop(columns=conf_matrix_2.columns[0], inplace=True)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_2, annot=True, cmap='coolwarm', fmt='.2f',xticklabels =conf_matrix.columns.tolist(), yticklabels =conf_matrix.columns.tolist() )
plt.title('Cosine Similarity Heatmap')
plt.xlabel('Hardware')
plt.ylabel('Hardware')
plt.show()