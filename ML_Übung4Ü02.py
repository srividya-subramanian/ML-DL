# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 18:27:24 2025

@author: srivi
"""
import pandas as pd
df = pd.DataFrame([
    ['Red'], ['Green'], ['Blue'], ['Red']
    ])
df.columns = ['Color']


def onehotcoding(df) :
    df_dum = pd.get_dummies(df)*1
    df_new = pd.concat([df,df_dum],axis=1)
    return df_new

print(onehotcoding(df))


from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse_output=False)
categorical_columns = ['Color']

df_pandas_encoded = pd.get_dummies(df, columns=categorical_columns)#, drop_first=True)
one_hot_encoded = enc.fit_transform(df[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=enc.get_feature_names_out(categorical_columns))
df_sklearn_encoded = pd.concat([df, one_hot_df], axis=1)

print(f"One-Hot Encoded Data using Scikit-Learn:\n{df_sklearn_encoded}\n")





import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Create a dummy employee dataset
data = {
    'Employee id': [10, 20, 15, 25, 30],
    'Gender': ['M', 'F', 'F', 'M', 'F'],
    'Remarks': ['Good', 'Nice', 'Good', 'Great', 'Nice']
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)
print(f"Original Employee Data:\n{df}\n")
# Use pd.get_dummies() to one-hot encode the categorical columns
df_pandas_encoded = pd.get_dummies(df, columns=['Gender', 'Remarks'], drop_first=True)
print(f"One-Hot Encoded Data using Pandas:\n{df_pandas_encoded}\n")

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
categorical_columns = ['Gender', 'Remarks']

# Fit and transform the categorical columns
one_hot_encoded = encoder.fit_transform(df[categorical_columns])

# Create a DataFrame with the encoded columns
one_hot_df = pd.DataFrame(one_hot_encoded, 
                          columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the one-hot encoded columns with the original DataFrame
df_sklearn_encoded = pd.concat([df.drop(categorical_columns, axis=1), one_hot_df], axis=1)

print(f"One-Hot Encoded Data using Scikit-Learn:\n{df_sklearn_encoded}\n")

print(onehotcoding(df))
