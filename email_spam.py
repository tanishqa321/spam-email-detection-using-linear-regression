#this project classifies email as spam and ham linear regression 
# here accuracy of training data is 96.61538461538461
#accuracy of testing data is  96.47129186602871
# then i have checked it on a spam mail from my gmail
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# we will do it in stages 1. loading an dpre processing data then splitting data into test and train here i have use 70% of data for training and 30 % for testing

# 1 Loading Data 
raw_mail_data = pd.read_csv('./mail_data.csv')
raw_mail_data.head()
# preprocessing the data by replacing null values by null string
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data),'')
#  categorising 0 as spam and 1 as ham
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Seperating the text as texts and label
X = mail_data['Message']
Y = mail_data['Category']
# print(X.head())
# print(Y.head())
# X-train have training data (messagges)  while Y_train have training data(categories for spam and ham)
X_Train,X_test,Y_Train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=3)
# print(X.shape)
# print(Y.shape)

#  Transform text data to feature vectors that can be used as input to the logistic regression
feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase= True)
#converting textual data into numerical  data using  TfidfVectorize
X_train_feature = feature_extraction.fit_transform(X_Train)
X_test_feature = feature_extraction.transform(X_test)

Y_Train = Y_Train.astype('int')
Y_test = Y_test.astype('int')

# print(X_train_feature)
# Training the Model
# we are using logistic regression 
#code fits (trains) the logistic regression model on the training data (X_train_feature) and corresponding target labels (Y_Train). Here,
# X_train_feature represents the features after feature extraction (assuming you've already performed feature extraction), and Y_Train
#  represents the corresponding target labels for the training set.

model = LogisticRegression()

model.fit(X_train_feature,Y_Train)


# Evaluating the Trained Model
# Predition on Training Model
prediction_on_Training_Data = model.predict(X_train_feature)
accuracy_on_training_data = accuracy_score(Y_Train,prediction_on_Training_Data)


print("Accuracy for Training : ",accuracy_on_training_data * 100)

# Predict on Test Data
prediction_on_Test_Data = model.predict(X_test_feature)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_Test_Data)
print("Accuracy for Testing : ",accuracy_on_test_data * 100)

#  Building a Predictable System
input_mail = ["Subject: Congratulations! You've Won $1,000,000! Dear Sir/Madam,We are pleased to inform you that you have been selected as the lucky winner of our annual lottery draw! You have won a grand prize of $1,000,000! To claim your prize, please reply to this email with your full name, address, phone number, and a copy of your identification. Don't miss out on this amazing opportunity! Act now to claim your winnings! Sincerely,Lottery Prize Department"]

# Convert Text to feature vectors
input_data_feature = feature_extraction.transform(input_mail)

# Making Prediction
prediction = model.predict(input_data_feature)

print(prediction)

if(prediction == [1]):
    print("This is the Ham Mail.")
else:
    print("This is the Spam Mail.")