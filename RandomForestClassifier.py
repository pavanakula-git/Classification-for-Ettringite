###################################################################
## Implementation of random forest classifier 					 ##
## Authors: Pavan Akula and Sandhya Saisubramanian 				 ##
###################################################################

import random
import numpy as np
import statistics
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterSampler
from subprocess import call

class Classifier:
	def __init__(self):
		self.params = {"min_samples_split": [2, 5, 10],
					   "max_depth": [None],
						"max_features":['auto'],
					   "min_samples_leaf": [1, 2],
					   "max_leaf_nodes": [None],
					   "n_estimators": [80, 100, 120, 200],
					   # "n_estimators": [10],
					   "random_state":[None,0],
					   "bootstrap":[True]}
		self.final_model = RandomForestClassifier()
		self.cls = RandomForestClassifier()


	#  Printing the decision tree
	def generate_tree(self):
		# Extract single tree
		estimator = self.final_model.estimators_[len(self.final_model.estimators_)-1]
		# Export as dot file
		export_graphviz(estimator, out_file='EttringiteClassifier.dot', 
				feature_names = ['pH','SS(%)','SiO2','Al2O3','CaO','SO3','Na2O','K2O'],
				class_names = ['yes','no'],
				rounded = True, proportion = False, 
				precision = 2, filled = True)

		# Convert to png using system command (requires Graphviz)
		call(['dot', '-Tpng', 'EttringiteClassifier.dot', '-o', 'EttringiteClassifier.png', '-Gdpi=600'])

	
	def train_folds(self,model,x_train_, y_train_, x_test_, y_test_): 
		model.fit(x_train_,y_train_)
		test_label = model.predict(x_test_)
		accuracy_folds = accuracy_score(test_label, y_test_)
		return accuracy_folds


	def Predict(self,x_train,y_train,x_test,y_test,testInput, processedFile,header,important_features_file):
		x=np.array(x_train)
		y=np.array(y_train)

		candidate_params = list(ParameterSampler(param_distributions=self.params, n_iter=10, random_state=None))
		model = clone(self.cls)
		num_splits = 3
		best_params = None
		best_score = -1

		# Loop through all possible parameter configurations and find the best one
		for parameters in candidate_params:
			model.set_params(**parameters)
			cv =  StratifiedKFold(n_splits=num_splits,random_state= None,shuffle=True)
			cv_scores  = []
		
			for train,test in cv.split(x, y):
				x_train_, x_test_, y_train_, y_test_ =  x[train], x[test], y[train], y[test]
				score = self.train_folds(model,x_train_,y_train_,x_test_,y_test_)
				cv_scores.append(score)
			avg_score = float(sum(cv_scores))/len(cv_scores)
			if(avg_score > best_score):
				best_params = parameters
				best_score = avg_score

		print(best_params)
		self.final_model = clone(model)
		self.final_model.set_params(**best_params)
		self.final_model.fit(x_train,y_train)

		important_features_dict = {}
		for x,i in enumerate(self.final_model.feature_importances_):
			important_features_dict[x] = i

		op_write = open(important_features_file,"a+")
		important_features_list = sorted(important_features_dict, key=important_features_dict.get, reverse=True)
		feature_names = ['pH','SS(%)','SiO2','Al2O3','CaO','SO3','Na2O','K2O']
		print('Most important features:')
		op_write.write("Most important features:\n")
		for f in important_features_list:
			print(feature_names[f], important_features_dict[f])
			op_write.write(str(feature_names[f])+","+str(important_features_dict[f])+"\n")

		op_write.write("*************************************\n")
		op_write.close()

		test_label = self.final_model.predict(x_test)
		predictions_all = np.array([tree.predict(np.array(x_test[5]).reshape(1,-1)) for tree in self.final_model.estimators_])
		accuracy = accuracy_score(y_test, test_label)
		print("Accuracy:", accuracy)
		myfile = open(processedFile,"w")
		myfile.write(header+",Predicted value"+"\n")
		for i in range(len(testInput)):
			f = ''.join(str(testInput[i]))
			f = f.replace('[','').replace(']','')
			myfile.write(f+",")
			myfile.write(str(test_label[i])+"\n")
		myfile.close()
		return accuracy
