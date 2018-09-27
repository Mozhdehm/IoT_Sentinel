IoT Sentinel project :
Device Type Identification by Random Forest Classification
Author: Mozhdeh Yamani  Mzhd.yamani@gmail.com




import numpy as np
import os,sys
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


input="IoT_CSVs"
all_files=sorted(os.listdir(input))
cpt=0

for item in all_files:
	data = np.genfromtxt(input+"/"+item, delimiter=',')	


    # set lables to 1 for targeted deviec 

	labels=[1]*(data.shape[0]) 

    # number of samples to pick from other devices

	s=int(round(float(data.shape[0])/28))  
	
	cpt=0
	aggr=None
	for f in all_files:
		if f!=item:
			rest=np.genfromtxt(input+"/"+f, delimiter=',')

			#random select a subset of all other device
			idx = np.random.randint(10, size=s)      
			rest=rest[idx,:]
			if cpt==0:
				aggr=rest
				cpt=cpt+1
			else:
				aggr=np.concatenate((aggr, rest), axis=0)

    # set lable to 0 for all other device

	labels_rest=[0]*aggr.shape[0]  

	 # make a train set for RF   
	x=np.concatenate((data, aggr), axis=0) 
	
	labels.extend(labels_rest)

	
	x_train,x_test,y_train,y_test = train_test_split(x,labels,test_size=0.3,random_state=42)
	rf=RandomForestClassifier(n_estimators=100,oob_score=True)
	rf.fit(x_train,y_train)
	predicted =rf.predict(x_test)
	accuracy = accuracy_score(y_test,predicted)
	print (item[:-4])
	print (rf.oob_score_)
	print (accuracy)
	pickle.dump(rf, open(item[:-4]+".pk", 'wb'))