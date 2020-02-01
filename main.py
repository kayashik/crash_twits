import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

#Load the dataset
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

train_df.columns = map(str.lower, train_df.columns)
print(train_df.columns)

test_df.columns = map(str.lower, test_df.columns)
print(test_df.columns)

predictors = train_df[['keyword','location','text']]
target = train_df.target

print("All done!")