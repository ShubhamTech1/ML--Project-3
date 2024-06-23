

# 1] FEATURE ENGINEERING:-

import pandas as pd 
df = pd.read_excel(r"D:\Off-Campus\cerina health\Homestays_Data.xlsx")

imp_features = ['amenities', 'host_since','last_review']   
df1 = df.loc[:, imp_features]  
# Host Tenure (Years):
from datetime import datetime 
# Assuming 'host_since' is a string in YYYY-MM-DD format
df1['host_since'] = pd.to_datetime(df1['host_since']) 
# Calculate current date
today = datetime.today()

# Create a new feature 'Host_Tenure' (years)
df1['Host_Tenure'] = (today.year - df1['host_since'].dt.year) + (today.month - df1['host_since'].dt.month) / 12
# Handle potential negative values (due to missing data) by setting them to 0 (assuming new hosts)
df1['Host_Tenure'] = df1['Host_Tenure'].clip(lower=0) 

# Amenities Count:    
# Assuming 'amenities' is a list/array within each data point
import json 
# Define a function to handle potential JSON parsing errors and count amenities
def count_amenities(amenities_str):
  try:
    amenities_list = json.loads(amenities_str.strip()[1:-1])
  except json.JSONDecodeError:
    amenities_list = amenities_str.strip()[1:-1].split(',')
  return len(amenities_list)
df1['Amenities_Count'] = df1['amenities'].apply(count_amenities) 

# Days Since Last Review:
# Assuming 'last_review' is also a string in YYYY-MM-DD format
df1['last_review'] = pd.to_datetime(df1['last_review']) 
df1['Days_Since_Last_Review'] = (today - df1['last_review']).dt.days



# 3] Geospatial Analysis:
    
imp_features1 = ['latitude', 'longitude','neighbourhood','log_price']  
df2 = df.loc[:, imp_features1] 

neighborhood_prices = {} 
for index, row in df2.iterrows():
  neighborhood = row['neighbourhood']  # Assuming you've assigned neighborhood by now
  price = df2['log_price'].iloc[index]  # Access price from original DataFrame
  if neighborhood not in neighborhood_prices:
    neighborhood_prices[neighborhood] = []
  neighborhood_prices[neighborhood].append(price)

# Calculate descriptive statistics (e.g., average price) for each neighborhood
for neighborhood, prices in neighborhood_prices.items():
  average_price = sum(prices) / len(prices)
  print(f"Neighborhood: {neighborhood}, Average Price: ${average_price:.2f}")


# 4. Sentiment Analysis on Textual Data:

df3 = pd.DataFrame(df.iloc[:, 11:-17])   # sentiment analysis on Description column.

from textblob import TextBlob   # library for sentiment score 

def get_sentiment_score(text):
  if not isinstance(text, str):
    return 0  # Assign neutral score for non-string values (modify as needed)
  text = text.lower().strip()  # Preprocess text
  blob = TextBlob(text)
  return blob.sentiment.polarity  

df3['sentiment_score'] = df3['description'].apply(get_sentiment_score)

# Range of sentiment scores:
print(df3['sentiment_score'].min(), df3['sentiment_score'].max())
# apply thresold value to check positive seentiment or negative sentiment
def classify_sentiment(score, threshold):
  if score > threshold:
    return 1  # Positive
  else:
    return 0  # Negative

df3['sentiment_category'] = df3['sentiment_score'].apply(lambda x: classify_sentiment(x, 0.2))
sentiment = pd.DataFrame(df3.iloc[2:, 2]) 
  

# 5. Amenities Analysis:

df4 = pd.DataFrame(df.iloc[2:, 4])   # independent feature -Amenities data 

df4["amenities"].unique()      
df4["amenities"].value_counts()   
 
# here we use bag  of words Technique to capture semntic meaning and convert words into vector.
from sklearn.feature_extraction.text import CountVectorizer
# Extract the amenities column
amenities_data = df4["amenities"] 
# Create a CountVectorizer object
vectorizer = CountVectorizer() 
# Fit the vectorizer on the amenities data (learn all possible amenities)
encoded_amenities = vectorizer.fit_transform(amenities_data)
# You can convert the sparse matrix to a DataFrame for easier handling (optional)
amenities_df = pd.DataFrame(encoded_amenities.toarray(), columns=vectorizer.get_feature_names_out())



# 6. Categorical Data Encoding:
    
imp_features2 = ['room_type', 'city', 'property_type']
df5 = df.loc[:, imp_features2] 
 # One-hot encode categorical features
df5 = pd.get_dummies(df5, columns=['room_type', 'city', 'property_type'])



























# 2 Exploratory Data Analysis (EDA):-

imp_features3 = ['log_price', 'accommodates', 'bathrooms', 'review_scores_rating','bedrooms','beds']
df6 = df.loc[:, imp_features3]  

# This features to play important role to predict log_price:-
new_df = pd.concat([sentiment, amenities_df, df5, df6], axis=1)   

# Correlation:-
correlation = new_df.corr() 
a = (correlation[['log_price']])

# SCATTER PLOT:-
import matplotlib.pyplot as plt 
import numpy as np 

for feature in new_df.columns:
  plt.scatter(new_df[feature], new_df['log_price'])
  plt.xlabel(feature)
  plt.ylabel('Log Price')
  plt.title(f'Scatter Plot: {feature} vs Log Price')
  plt.show()
# conclusion: independent features are not linearlly separable:

# Check data is normally distributed or not:-
import seaborn as sns 
sns.kdeplot(new_df["log_price"])   # Normally distributed
sns.kdeplot(new_df["shampoo"])    # Not Normally distributed

# check Linearity:-
# QQ Plot for 1st Feature:
import statsmodels.api as sm
plt.figure() 
sm.qqplot(new_df['log_price'], line='s')  # 's' for straight line

# 2nd feature
plt.figure() 
sm.qqplot(new_df['shampoo'], line='s')  # 's' for straight line

# TRANSFORMATION TECHNIQUE:- convert non linea feature into linear feature using log with base10 transformation. 
import numpy as np
import statsmodels.api as sm
log10_shampoo = np.log10(new_df['shampoo'])
sm.qqplot(log10_shampoo, line='s')



# 7. Model Development and Training:

new_df.shape   
new_df = new_df.drop_duplicates()
new_df.duplicated().sum()   # not any duplicated rows are present here.
# Seperating input and output variable.
log_price_column = 'log_price'  
X = new_df.drop(log_price_column, axis=1)  # independent features
Y = pd.DataFrame(new_df.iloc[:, -6])     # dependent features  


X.isnull().sum() # missing values are present
import seaborn as sns
X.plot(kind = 'box', subplots = True, sharey = False, figsize = (20, 8)) # here we see huge outliers are present here.

# Segregating Numeric features
numerical = X.select_dtypes(exclude = ['object']).columns


# 1] impute missing values:
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean'))]) 
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline,numerical)]) 
# Fit the imputation pipeline to input features
imputation = preprocessor.fit(X) 
# Transformed data
clean = pd.DataFrame(imputation.transform(X), columns = numerical)
clean.isnull().sum()           # successfully remove all null values.



# 2] retain the outliers:   
from feature_engine.outliers import Winsorizer
# Winsorization for outlier treatment
winsor = Winsorizer(capping_method = 'gaussian', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 0.20,
                          variables = list(clean.columns))

clean1 = winsor.fit(clean)
cleandata = pd.DataFrame(clean1.transform(clean), columns = numerical)
cleandata.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 
# successfully remove all the outliers


# 3]Feature Scaling:-
from sklearn.preprocessing import MinMaxScaler 
scale_pipeline = Pipeline([('scale', MinMaxScaler())]) 
columntransfer = ColumnTransformer([('scale', scale_pipeline, numerical)]) 
scale = columntransfer.fit(cleandata) 
scaled_data = pd.DataFrame(scale.transform(cleandata), columns = numerical)
scaled_data.isnull().sum() 




# MODEL BUILDING:- 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Split data into training and testing sets (optional but recommended)
X_train, X_test, y_train, y_test = train_test_split(scaled_data, Y, test_size=0.2, random_state=42)



# 1] LINEAR REGRESSION MODEL:-

model = LinearRegression()
# Train the model
model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

from sklearn.metrics import mean_squared_error
mse1 = mean_squared_error(y_test, y_predicted)
print("Mean Squared Error:", mse1) 
 



# 8]. Model Optimization and Validation:
    
from sklearn.model_selection import GridSearchCV, KFold 

# Define the linear regression model
model = LinearRegression() 

# Define the hyperparameter grid (without 'normalize')
param_grid = {
    'fit_intercept': [True, False],  # Whether to fit an intercept
}

# Create a KFold object (adjust k as needed)
kfold = KFold(n_splits=5, shuffle=True)  # 5 splits for cross-validation

# Perform GridSearchCV with KFold on scaled data
grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=kfold)
grid_search.fit(X_train, y_train) 

# Get the best model and mse
best_model = grid_search.best_estimator_
best_mse = grid_search.best_score_ * -1  # Convert negative MSE to positive

print("Best Mean Squared Error (GridSearchCV):", best_mse)

# Evaluate the best model on unseen test data (using scaled data)
y_predicted = best_model.predict(X_test)

# Calculate final MSE on the test set
final_mse = mean_squared_error(y_test, y_predicted)

print("Final Mean Squared Error on Test Set:", final_mse)





# 2] ANN MODEL:
    
from tensorflow import keras  # Assuming you have TensorFlow installed

# Define the ANN model
model1 = keras.Sequential([
    keras.layers.Dense(units=32, activation="relu", input_shape=(X_train.shape[1],)),  # First hidden layer with 32 neurons and relu activation
    keras.layers.Dense(units=16, activation="relu"),  # Second hidden layer with 16 neurons and relu activation
    keras.layers.Dense(units=1)  # Output layer with 1 neuron for regression
])

# Compile the model (specifies optimizer and loss function)
model1.compile(optimizer="adam", loss="mse", metrics = ['accuracy'])  # Adam optimizer and mean squared error loss for regression

# Train the model
model1.fit(X_train, y_train, epochs=15, batch_size=32)  # Train for 100 epochs with batches of 32 samples

# Predict on the test set
y_predicted = model1.predict(X_test)

# Calculate mean squared error
from sklearn.metrics import mean_squared_error
mse2 = mean_squared_error(y_test, y_predicted)
print("Mean Squared Error:", mse2)
    





# 3] RANDOM FOREST:-

from sklearn.ensemble import RandomForestRegressor

# Define the Random Forest model
model2 = RandomForestRegressor(n_estimators=100, random_state=42)  # Adjust n_estimators as needed

# Train the model
model2.fit(X_train, y_train) 

# Predict on the test set
y_predicted = model2.predict(X_test)

# Calculate mean squared error
from sklearn.metrics import mean_squared_error
mse3 = mean_squared_error(y_test, y_predicted)
print("Mean Squared Error:", mse3)






# 4] XG-Boost:-

# pip install xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error 

# Define XGBoost model
model3 = XGBRegressor(objective='reg:squarederror', max_depth=5, n_estimators=100)  # Adjust hyperparameters as needed

# Train the model
model3.fit(X_train, y_train)

# Make predictions
y_predicted = model3.predict(X_test)

# Evaluate the model
mse4 = mean_squared_error(y_test, y_predicted)
print("Mean Squared Error:", mse4)



# 9]. Feature Importance and Model Insights:
    
from sklearn.ensemble import RandomForestRegressor  # Example tree-based model
from sklearn.inspection import permutation_importance
#pip install shap 
import shap

# Assuming you have trained models (model1, model2, etc.) and data (X_train, X_test, y_train, y_test)
# Feature Importance for Tree-based Model (e.g., RandomForestRegressor)
def analyze_feature_importance(model, X_test, y_test, feature_names):
  results = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
  for name, importance in zip(feature_names, results.importances_mean):
      print(f"Feature: {name}, Importance (Tree-Based): {importance:.4f}")

# Analyze SHAP Values for any Model
def analyze_shap(model, X_train, X_test, feature_names):
  explainer = shap.Explainer(model, X_train)
  shap_values = explainer(X_test)
  shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Example Usage (replace with your actual models)
# Analyze feature importance for a Random Forest model
analyze_feature_importance(RandomForestRegressor().fit(X_train, y_train), X_test, y_test, X_test.columns)

# Analyze SHAP values for model1 (replace with your model)
analyze_shap(model3, X_train, X_test, X_test.columns) 




# 10] Predictive Performance Assessment:
    
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a final model (model), reserved test set (X_test, y_test), and predicted values (y_predicted)

# Evaluation Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
print("Root Mean Squared Error (RMSE):", rmse)

r2 = r2_score(y_test, y_predicted)
print("R-squared:", r2)

# Residual Analysis
residuals = y_test - y_predicted
residuals
# Residuals vs True Values plot
plt.scatter(y_test, residuals)
plt.xlabel("True Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. True Values")
plt.show()

# Distribution of Residuals plot
plt.hist(residuals)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.show()
   

    
