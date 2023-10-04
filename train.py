import pandas as pd
import matplotlib as plt
from sklearn.model_selection import cross_val_score , train_test_split
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import r2_score

df=pd.read_csv('C:/Users/DELL/Desktop/house_prices/Housing.csv')
df
Ycol = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
i=0
for col in Ycol:
    df[col]=df[Ycol[i]].replace({'yes':1,'no':0},)
    i+=1
df    
NDF=pd.get_dummies(df.furnishingstatus)
NDF
df=pd.concat([df,NDF],axis=1)
df
df.drop(['furnishingstatus'],axis=1,inplace=True)
df
df.drop(['unfurnished'],axis=1,inplace=True)
df
x=df.drop(['price'],axis=1)
x
y=df.price
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
def Train_Test(model,model_name):
    model.fit(x,y)
    model_train_score=model.score(x_train,y_train)
    model_test_score=model.score(x_test,y_test)
    print(f"{model_name} model score on Training data: {model_train_score * 100}%\n{model_name} model score on Testing data: {model_test_score * 100}%")
    return model

def R2(model, model_name):
    acc=r2_score(y_test,model.predict(x_test))
    print(f"R2 Score for {model_name} is {acc * 100}%")
model=linear_model.LinearRegression()
Train_Test(model,'linear regerssion')
R2(model,'linear regerssion')
model_rf = RandomForestRegressor()
model_rf = Train_Test(model_rf,'Random forest')
# Get the R2 score for the random forest model
random_r2 = R2(model_rf,'random forest')

# open file to write the model
with open('house_prices.pkl', 'wb') as file:
    # dump the model object into the file
    pickle.dump(model_rf, file) #put model in file

# print confirmation message
print("Trained model saved successfully!",file)