###################################################################
## Master code to test prediciton with classifier                ##
## Authors: Pavan Akula and Sandhya Saisubramanian 				 ##
###################################################################

import numpy as np
import matplotlib.pyplot as plt
from RandomForestClassifier import *
import sys

import random 

def main():
	print("********************* Methodology for prediction:")
	print(" 1. classification \n 2. Multi-run classification")
	val = input("Predict with methodology:")
	if int(val) not in [1,2,3,4]:
		print("Invalid entry")
		sys.exit(1)

	if int(val) == 1:
		run_classification()
	elif int(val) == 2:
		generateRandomTrainTest_classifier()

def readFile(filename):
	myfile = open(filename,"r")
	to_skip_index = [0,6,7]
	line_index = -1
	x= []
	y = []
	instances = []
	header = []
	for line in myfile:
		line_index += 1
		if line_index == 0:
			header = line
			continue
		val = line.split(",")
		y.append(round(float(val[len(val)-1])))
		data = []
		all_data = []
		for d in range(len(val)-1):
			if d in to_skip_index:
				continue
			data.append(float(val[d]))
		for d in range(len(val)):
			all_data.append(float(val[d]))

		x.append(data)
		instances.append(all_data)
	myfile.close()
	return x,y,instances,header.strip()


def plotPrediction(filename):
	myfile = open(filename)
	true_val = []
	predicted_val = []
	line_index = -1
	for line in myfile:
		line_index += 1
		if line_index == 0:
			continue
		val = line.split(",")
		true_val.append(float(val[len(val)-2]))
		predicted_val.append(float(val[len(val)-1]))
	myfile.close()
	error_bar_upper = [10+true_val[i] for i in range(len(true_val))]
	error_bar_lower = [true_val[i]-10 for i in range(len(true_val))]
	N = np.arange(len(true_val))
	plt.scatter(N,true_val, color="green",label="true_val")
	plt.scatter(N,predicted_val,color="blue",label="predicted_val", marker="*")
	plt.plot(error_bar_upper,color="black")
	plt.plot(error_bar_lower,color="black")
	plt.legend()
	plt.xlabel("Instances")
	plt.ylabel("Ettringite %")
	plt.show()


def run_classification():
	x_train, y_train, full_train_data, header = readFile("Ett_train_classification_1.csv")
	x_test, y_test, full_test_data, header =  readFile("Ett_test_classification_1.csv")
	output_file = "EttPrediction_classifier_1.csv"
	important_features_file = "Important_features_classifier.txt"
	rc = Classifier()
	rc.Predict(x_train,y_train,x_test,y_test,full_test_data, output_file, header,important_features_file)
	# rc.generate_tree()



def generateRandomTrainTest_classifier():
	random.seed(100)
	x,y,full_data, header = readFile("Ett_classifier_fulldata.csv")
	num_iter = 5
	accuracy_arr = np.zeros((num_iter))
	important_features_file = "Important_features_classifier_multirun.txt"
	
	for i in range(num_iter):
		x_train = []
		y_train = []
		x_test = []
		y_test = []
		full_test_data = []

		output_file = "EttPrediction_classifier_"+ str(i+1) +".csv"
		training_instances = random.sample(x,int(round(0.8 * len(x))))
		for v in range(len(x)):
			if x[v] in training_instances:
				x_train.append(x[v])
				y_train.append(y[v])
			else:
				x_test.append(x[v])
				y_test.append(y[v])
				full_test_data.append(full_data[v])
		rc = Classifier()
		accuracy = rc.Predict(x_train,y_train,x_test,y_test,full_test_data, output_file, header,important_features_file)
		accuracy_arr[i] = accuracy
	
	avg_accuracy = np.average(accuracy_arr)
	std = np.std(accuracy_arr,axis=0)
	print("Average accuracy = %s, standard deviation = %s"%(avg_accuracy,std))


if __name__ == "__main__":
	main()


