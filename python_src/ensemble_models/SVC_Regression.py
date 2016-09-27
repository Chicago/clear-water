######################################################
#### SVC Model
######################################################

import pandas as pd
import statsmodels.api as sm
from sklearn import svm,preprocessing
from sklearn.externals import joblib

data_processed = pd.DataFrame.from_csv('data_processed.csv')
meta_info = pd.read_csv('meta_info.csv')

for i in range(len(meta_info['Escherichia.coli'])) :
	if meta_info.loc[i,'Escherichia.coli'] < 236:
		meta_info.loc[i,'E_coli']= int(0)
	else :
		meta_info.loc[i,'E_coli']= int(1)


E_coli = meta_info['E_coli'].values.tolist()
test_size = 1000
X=preprocessing.scale(data_processed)
svc= svm.SVC(kernel='poly', degree=3, C=100.0, gamma=0.07)
svc.fit(X[:-test_size],E_coli[:-test_size])

joblib.dump(svc, 'C:/Users/Callin/Documents/GitHub/Chicago/e-coli-beach-predictions/python_src/ensemble_models/models/svc_clf.pkl', compress=9)

prediction=[]
correct_count = 0
for x in range(1,test_size+1):
	prediction.append(svc.predict(X[[-x]])[0])

fp = 0
fn = 0		
for i in range(1,test_size+1):
	if prediction[i-1] != E_coli[-i]:
		if prediction[i-1] == 1:
			fp+=1
		else :
			fn +=1

print("TotalCorrect: ",sum(prediction)-fp,"False Positive:", fp, "False Negative: ", fn, "Total Called: ", sum(prediction), "Precision: ", (sum(prediction)-fp)/(fn+(sum(prediction)-fp)))	
