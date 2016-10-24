######################################################
#### SVC Model
######################################################

import os
import pandas as pd
import statsmodels.api as sm
from sklearn import svm,preprocessing
from sklearn.externals import joblib

cwd = os.getcwd() #Find the current working directory

data_processed = pd.DataFrame.from_csv(cwd+'\\data_processed.csv') #Read in Data_processed.csv
meta_info = pd.read_csv(cwd+'\\meta_info.csv') #Read in Meta_processed.csv

#Find where the E.coli was high(above 235ppm)
for i in range(len(meta_info['Escherichia.coli'])) :
	if meta_info.loc[i,'Escherichia.coli'] < 236:
		meta_info.loc[i,'E_coli']= int(0)
	else :
		meta_info.loc[i,'E_coli']= int(1)


E_coli = meta_info['E_coli'].values.tolist() #E.coli is just a 1 or 0 to signify high or low to train the SVC.
test_size = 1000 #How many rows to predict on.
X=preprocessing.scale(data_processed) #Trim the variation down
svc= svm.SVC(kernel='poly', degree=3, C=20, gamma=0.07)#Create the SVC model
svc.fit(X[:-test_size],E_coli[:-test_size]) #Train the SVC model


joblib.dump(svc, cwd+'\\models\\svc_clf.pkl', compress=9) #create a .pkl file to be able to use in other programs

prediction=[] #A variable to store the predictions

for x in range(1,test_size+1):
	prediction.append(svc.predict(X[[-x]])[0])

fp = 0 #Variable to count false positives
fn = 0 #Variable to count false negatives

#Check to see if the prediction and reality match, if they do add 1 to fp, otherwise
for i in range(1,test_size+1):
	if prediction[i-1] != E_coli[-i]:
		if prediction[i-1] == 1:
			fp+=1
		else :
			fn +=1

#Print some simple stats to see how the SVC Model did.
print("TotalCorrect: ",sum(prediction)-fp,"False Positive:", fp, "False Negative: ", fn, "Total Called: ", sum(prediction), "Precision: ", (sum(prediction)-fp)/(fn+(sum(prediction)-fp)))	
