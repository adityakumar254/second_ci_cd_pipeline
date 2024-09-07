import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('data/iris.csv')

# preprocess the dataset
x= data.drop('species', axis=1)
y= data['species']


# Split the data into traning an test sets 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# load the saved model 
model = joblib.load('model/iris_model.pkl')

# make predictions
y_pred = model.predict(x_test)


# Evaluate the model 
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:2f}')
 
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')