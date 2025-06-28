import numpy as np #help in handling array
import pandas as pd #to make dataframe
import matplotlib.pyplot as plt #to plot graph
import seaborn as sns #also use for making some plot
from sklearn.model_selection import train_test_split #train and test all the data
from sklearn.ensemble import RandomForestRegressor #
from sklearn import metrics #to accuracy of model


#loading the csv data  to a pandas dataframe

gold_data = pd.read_csv('gld_price_data.csv')

#print first 5 rows in the dataframe
print(gold_data.head())

#print last 5 rows in the dataframe
print(gold_data.tail())

#number for rows and column

print(gold_data.shape)

#getting some basic information about the data

print(gold_data.info())

#checking the null value in dataframe

print(gold_data.isnull().sum())

#getting statistical measures of the data 

print(gold_data.describe())

#to to check correlation
'''
and as we have the date format in the csv file so the The error occurs because the corr() function in pandas is designed to 
calculate the correlation between numerical columns. The 'Date' column is of string type (object dtype), 
which cannot be directly used in correlation calculations.
'''
correlation = gold_data.drop('Date', axis=1).corr()

# constructing a heatmap to understand the correlation

plt.figure(figsize = (8,8))
sns.heatmap(correlation, annot=True, cbar=True, square = True,fmt = '.1f'  ,annot_kws={'size': 8}   , cmap='Blues')
plt.show()

#correlation values of GLD
print(correlation['GLD'])

# check to distribution of the GLD price
sns.displot(gold_data['GLD'], color='green')
plt.show()

# splitting the features and Target
x = gold_data.drop(['Date','GLD'],axis = 1 )
y = gold_data['GLD']

#now sepreating x and y into test data and training data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3) 

#model training : Random forest regressor
regressor = RandomForestRegressor(n_estimators=100)

#training the model
regressor.fit(x_train, y_train)
print(regressor)



#prediction on test data

test_data_prediction = regressor.predict(x_test)
print(test_data_prediction)

# Evaluate the model
r2 = metrics.r2_score(y_test, test_data_prediction)
print(f"R² Score: {r2:.4f}")

#comparing the actual value and the predicted value 

y_test = list()
plt.plot(y_test, color='blue', label='acutal value')
plt.plot(test_data_prediction, color='green', label='predicted value')

plt.title('actual value vs predicted value')
plt.xlabel('number of values')
plt.ylabel('gold price')
plt.legend()
plt.show()
