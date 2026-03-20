
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# LOAD DATA
df = pd.read_csv("student_habits_performance.csv")

# Drop unnecessary column
df.drop(columns=['student_id'], inplace=True)

# SPLIT
X = df.drop(columns=['exam_score'])
y = df['exam_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# FEATURE GROUPS
num_cols = [
    'age','study_hours_per_day','social_media_hours',
    'netflix_hours','sleep_hours','exercise_frequency','mental_health_rating'
]

binary_cols = ['part_time_job', 'extracurricular_participation']
nominal_cols = ['gender']

ordinal_cols = [
    'diet_quality',
    'parental_education_level',
    'internet_quality'
]

# ORDERS
diet_order = ['Poor', 'Fair', 'Good']
education_order = ['High School', 'Bachelor', 'Master']
internet_order = ['Poor', 'Average', 'Good']

# PIPELINES
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

binary_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='if_binary'))
])

nominal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[
        diet_order,
        education_order,
        internet_order
    ]))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('bin', binary_pipeline, binary_cols),
    ('nom', nominal_pipeline, nominal_cols),
    ('ord', ordinal_pipeline, ordinal_cols)
])

# FINAL PIPELINE
model_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', LinearRegression())
])

# TRAIN
model_pipeline.fit(X_train, y_train)

# SAVE MODEL
with open("model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("Mark_Predictor_Model.pkl")