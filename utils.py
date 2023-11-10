
## main libraries
import numpy as np
import pandas as pd
import os

## sklearn (preprocessing)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder ##,RobustScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.impute import SimpleImputer, KNNImputer


## Define the path to the train.csv file
TRAIN_PATH = os.path.join(os.getcwd(), "train.csv")
## Read the train.csv file into a DataFrame, using "Unnamed: 0" column as the index
df_train = pd.read_csv(TRAIN_PATH, index_col="Unnamed: 0")
## Define the path to the test.csv file
TEST_PATH = os.path.join(os.getcwd(), "test.csv")
## Read the test.csv file into a DataFrame, using "Unnamed: 0" column as the index
df_test = pd.read_csv(TEST_PATH, index_col="Unnamed: 0")
## Concatenate the train and test DataFrames vertically to create a combined DataFrame
df = pd.concat([df_train, df_test], axis=0)

## Feature selected by filter method and target
df = df[['Flight Distance', 'Arrival Delay in Minutes', 'Age', 'Customer Type', 'Type of Travel', 'Class', 'Leg room service', 'Food and drink', 'Online boarding', 'Baggage handling',
         'On-board service', 'Inflight wifi service', 'Ease of Online booking', 'Cleanliness', 'Inflight entertainment', 'Inflight service', 'Seat comfort', 'Checkin service', 'satisfaction']]

## encode target
df["satisfaction"] = df["satisfaction"].map({"neutral or dissatisfied":0, "satisfied":1})

## split data 
X = df.drop(columns=["satisfaction"], axis=1)
y = df["satisfaction"] ## target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42, stratify=y)

## store the column names for categorical.
categorical_cols =X.select_dtypes(include="O").columns.to_list()
## list of column names representing numerical.
numerical_cols = ["Flight Distance", "Arrival Delay in Minutes", "Age"]
# Calculate the set of columns that are ready for data processing, which excludes categorical and numerical columns.
ready_cols = list(set(X.columns.tolist()) - set(categorical_cols) - set(numerical_cols))


## Define a pipeline for processing numerical columns.
numerical_pipe = Pipeline(
    steps=[
        ("Selector", DataFrameSelector(numerical_cols)),  ## Select numerical columns
        ("impute", SimpleImputer(strategy="median")),     ## Impute missing values with median
        ("Transformer", FunctionTransformer(np.log1p)), ## using log transform
        ("Scaler", StandardScaler())                     ## Scale the numerical features ## RobustScaler
    ])

## Define a pipeline for processing categorical columns.
categorical_pipe = Pipeline(
    steps=[
        ("Selector", DataFrameSelector(categorical_cols)),        ## Select categorical columns
        ("impute", SimpleImputer(strategy='most_frequent')),     ## Impute missing values with most frequent value
        ("Encoding", OneHotEncoder(drop='first', sparse_output=False))  # One-hot encode categorical features
    ])

## Define a pipeline for processing columns that are "ready" for processing.
ready_pipe = Pipeline(
    steps=[
        ("Selector", DataFrameSelector(ready_cols)),  ## Select columns that are ready for processing
        ("impute", KNNImputer(n_neighbors=5)) ## Impute missing values using K-nearest neighbors
    ])

## Combine the numerical, categorical, and ready processing pipelines into one.

all_pipeline = FeatureUnion(
    transformer_list=[
        ('numerical', numerical_pipe),
        ('categorical', categorical_pipe),
        ('ready', ready_pipe)
    ])

## fit the training data using the combined pipeline.
X_train_final = all_pipeline.fit(X_train)




def new_process(new_sample):
    """
    Transform a new sample using a predefined pipeline.

    Parameters:
    - new_sample (list): list containing features of a new sample --> user input.

    Returns:
    - X_transformed (numpy.ndarray): Transformed features of the new sample.
    """

    # Convert to DataFrame
    df_new = pd.DataFrame([new_sample])
    df_new.columns = X.columns  # Assuming X is a global variable representing feature columns

    # NUMERICAL FEATURES
    df_new["Flight Distance"] = df_new["Flight Distance"].astype("int64")
    df_new["Arrival Delay in Minutes"] = df_new["Arrival Delay in Minutes"].astype("float64")
    df_new["Age"] = df_new["Age"].astype("int64")
    
    # CATEGORICAL FEATURES
    df_new["Customer Type"] = df_new["Customer Type"].astype("object")
    df_new["Type of Travel"] = df_new["Type of Travel"].astype("object")
    df_new["Class"] = df_new["Class"].astype("object")

    # READY FEATURES
    df_new["Ease of Online booking"] = df_new["Ease of Online booking"].astype("int64")
    df_new["Leg room service"] = df_new["Leg room service"].astype("int64")
    df_new["Online boarding"] = df_new["Online boarding"].astype("int64")
    df_new["Inflight service"] = df_new["Inflight service"].astype("int64")
    df_new["Inflight wifi service"] = df_new["Inflight wifi service"].astype("int64")
    df_new["Food and drink"] = df_new["Food and drink"].astype("int64")
    df_new["Inflight entertainment"] = df_new["Inflight entertainment"].astype("int64")
    df_new["Cleanliness"] = df_new["Cleanliness"].astype("int64")
    df_new["On-board service"] = df_new["On-board service"].astype("int64")
    df_new["Baggage handling"] = df_new["Baggage handling"].astype("int64")
    df_new["Seat comfort"] = df_new["Seat comfort"].astype("int64")
    df_new["Checkin service"] = df_new["Checkin service"].astype("int64")

    # Assuming 'all_pipeline' is a global variable representing the transformation pipeline
    X_transformed = all_pipeline.transform(df_new)
    return X_transformed



