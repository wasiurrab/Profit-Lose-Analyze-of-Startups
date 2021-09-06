import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

data=pd.read_csv('50_Startups.csv')
data.head()

def convert_to_int(word):
    word_dict = {'New York':1, 'California':2, 'Florida':3}
    return word_dict[word]

data['State'] = data['State'].apply(lambda x : convert_to_int(x))

X=data.drop(['Profit'],axis=1)
y=data['Profit']


regressor=LinearRegression()
regressor.fit(X,y)
pickle.dump(regressor, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[16985, 300000, 50000,2]]))
