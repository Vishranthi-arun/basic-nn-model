# Developing a Neural Network Regression Model
## AIM
To develop a neural network regression model for the given dataset.
## THEORY
Neural networks consist of simple input/output units called neurons. These units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.
<br><br>
Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.
<br><br>
Build your training and test set from the dataset, here we are making the neural network 2 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.
<br>
## Neural Network Model
![image](https://user-images.githubusercontent.com/93427278/226187960-957431be-e9fb-4907-81f6-a2bb20a34bea.png)
## DESIGN STEPS
### STEP 1:
Loading the dataset
### STEP 2:
Split the dataset into training and testing
### STEP 3:
Create MinMaxScalar objects ,fit the model and transform the data.
### STEP 4
Build the Neural Network Model and compile the model.
### STEP 5:
Train the model with the training data.
### STEP 6:
Plot the performance plot
### STEP 7:
Evaluate the model with the testing data.
## PROGRAM
```
Developed by: Vishranthi A
Register No. : 212221230124
```
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('vishu dataset').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})
df
import pandas as pd
from sklearn.model_selection import train_test_split
# To scale
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
X = df[['INPUT']].values
y = df[['OUTPUT']].values
X
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.33,random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)model = Sequential([
    Dense(5,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])
model.compile(optimizer='rmsprop',loss = 'mse')
model.fit(X_train1,y_train,epochs=5000)
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
model.evaluate(X_test1,y_test)
model.evaluate(X_test1,y_test)
## new prediction
X_n1 = [[500]]
X_n1_1 = Scaler.transform(X_n1)
model.predict(X_n1_1)
```
## Dataset Information
![image](https://user-images.githubusercontent.com/93427278/225969736-1cd238c3-730a-451a-ae09-d6f1df3d5c8c.png)
## OUTPUT
### Training Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/93427278/225967668-85eb8c4b-11e1-4d3c-926c-9041a5df9734.png)
![image](https://user-images.githubusercontent.com/93427278/226158533-41ac451f-e6d4-4266-a36d-cd3819af0129.png)
### Test Data Root Mean Squared Error
![image](https://user-images.githubusercontent.com/93427278/226158601-60d55de3-8047-4e53-9f54-655fe8a5dc6d.png)
### New Sample Data Prediction
![image](https://user-images.githubusercontent.com/93427278/226158628-3c45d909-0ad5-4721-abe3-a5dc825b184a.png)
## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully.
