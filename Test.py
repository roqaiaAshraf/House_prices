import pandas as pd
import matplotlib as plt
from sklearn.model_selection import  train_test_split
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
import pickle
with open('C:/Users/DELL/Desktop/house_prices/house_prices.pkl', 'rb') as file:
    # load the model object from the file
    model_rf = pickle.load(file)

# Define the input values for the house features
area = 99000
bedrooms = 4
bathrooms = 4
stories = 4
mainroad = 1
guestroom = 1
basement = 1
hotwaterheating = 1
airconditioning = 1
parking = 4
prefarea = 1
furnished = 1
semiFurnished = 0  # furnished=0 semiFurnished=0 >> it's unfurnished
input_data = [[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnished, semiFurnished]]
predicted_price = model_rf.predict(input_data)

# Print the predicted price of the house
print(f"The predicted price of the house is {predicted_price[0]}")