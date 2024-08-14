# task-6
Apply advanced statistical and analytical methods to solve complex problems.

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('your_timeseries_data.csv', parse_dates=True, index_col='Date')

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Time Series Data')
plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(data, model='additive')
decomposition.plot()
plt.show()
from statsmodels.tsa.arima.model import ARIMA

# Fit the model (ARIMA example)
model = ARIMA(data, order=(1, 1, 1))
results = model.fit()

# Forecasting
forecast = results.forecast(steps=12)  # Forecast for 12 periods
plt.plot(forecast)
plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('your_text_data.csv')

# Preprocessing
data['text_cleaned'] = data['text'].str.lower().str.replace('[^\w\s]','')

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(data['text_cleaned'], data['label'], test_size=0.2)
# Convert text to features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict and evaluate
predictions = model.predict(X_test_vec)
print(f'Accuracy: {accuracy_score(y_test, predictions)}')
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
data = pd.read_csv('your_clustering_data.csv')
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply K-Means
kmeans = KMeans(n_clusters=3)
kmeans.fit(data_scaled)

# Assign clusters
data['cluster'] = kmeans.labels_

# Plotting the clusters
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
plt.show()
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load and split the data
data = pd.read_csv('your_classification_data.csv')
X_train, X_test, y_train, y_test = train_test_split(data.drop('label', axis=1), data['label'], test_size=0.2)

# Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
