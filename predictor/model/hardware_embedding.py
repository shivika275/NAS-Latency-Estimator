import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import json
from sklearn.metrics.pairwise import cosine_similarity


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


with open('../system_info.json', 'r') as file:
    hardware_data = json.load(file)

# Convert JSON data to DataFrame
df = pd.DataFrame(hardware_data)

# Columns to one-hot encode
categorical_cols = ['machine', 'processor','gpu_name']

# Columns for feature scaling
numerical_cols = ['cores', 'threads', 'freq', 'max_freq', 'min_freq', 'ram_memory','gpu_memory_total','total_cores']

# Define preprocessing steps for columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Create a pipeline for preprocessing
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit the pipeline on the hardware data
embedded_data = pipeline.fit_transform(df)
cos_sim_matrix = cosine_similarity(embedded_data, embedded_data)
conf_matrix = pd.DataFrame(cos_sim_matrix, columns=[hardware_data[i]['machine'] for i in range(len(hardware_data))], 
                           index=[hardware_data[i]['machine'] for i in range(len(hardware_data))])

conf_matrix.to_csv('cosine_sim.csv')
# 'embedded_data' now contains the embeddings for hardware characteristics
# print("Hardware Embeddings:")
# print(embedded_data)
