import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from scipy.stats import t
import sys
import math

stepforqueues=10


count = 0
countneg = 0

risultatiPuliti= "../risultatiPuliti.csv"
risultatiSporchi= "../risultatiSporchi.csv"
graph= "../graph/"


attrDict = {
"customerQueueQ1":["queueLength"],
"energyQueueQ2":["queueLength"],
"sinkC":["lifeTime"]
}

paramsDict={
  "lambda":["2s", "4s"],
  "w":["1s", "2s", "4s"],
  "N":["40", "60", "100"],  #energia
  "K":["2","4","8","1"],   #customer -1 = infinita capacità della coda
  "p":["1","2","3"],
  "z":["0.04s", "0.058s", "0.07s"]
}

def iterateOnParams(list):
	if len(list)>0:
		for e in range(len(paramsDict[list[0]])):
			if len(list[1:])>0:
				for i in iterateOnParams(list[1:]):
					yield (e,*i)
			else:
				yield([e])
	else:
		return ([])

def printGraph(x, y, title, label="graph", descrX="x", descrY="y"):
	plt.plot(x,y, label=label)
	plt.plot(x,y, 'ro')
	plt.xlabel(descrX)
	plt.ylabel(descrY)
	plt.title(title)
	plt.savefig(graph+title.replace(" ", "_") + ".png")
	plt.show()

def printGraphOnTime(y, title, label="graph", descrX="x", descrY="y"):
	plt.plot(y, label=label)
	plt.plot(y, 'ro')
	plt.xlabel(descrX)
	plt.ylabel(descrY)
	plt.title(title)
	plt.savefig(graph+title.replace(" ", "_") + ".png")
	plt.show()


def printGraphP(x, title, label="graph", descrX="x", descrY="y"):
	plt.plot(x, label=label)
	plt.xlabel(descrX)
	plt.ylabel(descrY)
	plt.title(title)
	plt.show()

def printGraphWConf(x, y, confL, confU, title, label="graph", descrX="x", descrY="y"):
	fig ,ax = plt.subplots()
	ax.plot(x, y, label=label)
	ax.fill_between(x, confL, confU, facecolor='yellow')
	ax.set_xlabel(descrX)
	ax.set_ylabel(descrY)
	ax.set_title(title)
	plt.show()

def printMultiGraph(multi, title, descrX="x", descrY="y"):
	labels=[]
	ss=[]
	for x,y,s in multi:
		tmp,=plt.plot(x,y, label=s)
		labels.append(tmp)
		ss.append(s)
	plt.xlabel(descrX)
	plt.ylabel(descrY)
	plt.title(title)
	plt.legend(labels,ss)
	plt.show()

def calculateTransient(array):
	step = math.ceil(np.multiply(np.true_divide(len(array), 100), 2))
	max_diff = 0
	max_index = -1
	for i in range(step * 2, len(array), step):
		diff = abs(np.mean(array[0:i]) - np.mean(array[0:i - step]))
		if np.greater(diff, max_diff):
			max_diff = diff
			max_index = i
	return max(max_index, 0)


def deleteNvalues(array,value):
	array=array[value:]

def rowToStr(row):
	str1 = ''.join(str(e) for e in row)
	return(str1.replace("]","").replace("[","").replace(",",""))

def checkArgs():
	if (len(sys.argv)!=2):
		print("Insert name of project to analyze")
		exit(0)

def deletePrevResults(risultatiSporchi, risultatiPuliti):
	try:
		os.remove(risultatiSporchi)
		os.remove(risultatiPuliti)
	except OSError:
		pass

'''
def deletePrevGraph():
	try:
		for file in os.listdir(graph):
			os.remove(file)
	except Exception as e:
		print(e)
'''

#Create single dict for each configuration
def assembleDictionary(total):
	path = "./samples/"+sys.argv[1]+"/results/";
	files = os.listdir(path)
	for file in files:
		if file.endswith('.csv'):
			nameFile = file.split("#")[0]
			print("namefile = ",nameFile)
			run = file.split("#")[1].split(".")[0]
			csv = pd.read_csv(path+file)
			modules = csv.module.unique() #all modules
			attributes = csv.name.unique()
			type = csv.type.unique()
			params = nameFile[7:].replace('-','').split('-')[0].split(',')
			actual = total
			for p in params:
				p0 = p.split("=")[0].strip()
				p1 = p.split("=")[1].strip()
				index = paramsDict[p0].index(p1)
				actual = actual[index]
			for module in modules:
				if str(module)!="nan":
					moduleName = str(module).split(".")[1]
					if(len(actual[moduleName])) == 0:
						actual[moduleName] = defaultdict(list)
					if  moduleName in attrDict.keys():  #interesting modules
						print("MODULE:",str(moduleName))
						for attribute in attributes:
							if (str(attribute)!= "nan") :
								attributeName=str(attribute).split(":")[0]
								if (attributeName in attrDict[moduleName]):
									row = csv[(csv.type == 'vector') & (csv.module == module) & (csv.name == attribute)].vecvalue.describe()
									rowtime = csv[(csv.type == 'vector') & (csv.module == module) & (csv.name == attribute)].vectime.describe()
									if len(row.values) > 2:
										print("attr = ",attribute)
										rt = [(float(a),float(b)) for a,b in zip(row[2].split(), rowtime[2].split())]
										actual[moduleName][attributeName].append(rt)
										with open(risultatiSporchi,"a") as f:
											r = rowToStr(row[2])
											t = rowToStr(rowtime[2])
											f.write("Configuration, "+nameFile +", Module," +module + ", Attribute," +attribute +  ", Run, "+run+", Values, " + r + ", Time, " + t + "\n")

def createMeanArray(matrix,start=0,step=1):
	def calcMeanloc(row,start,end):
		acc=0
		counter=0
		for value, time in row:
			if start < time < end:
				acc=acc+value
				counter=counter+1
		if counter>0:
			return [acc/counter]
		else:
			return []

	arrayMean = []
	for interval in range(start,299,step):
		mean=[]
		for row in matrix:
			val=calcMeanloc(row,interval,interval+step) 
			if len(val)>0:
				mean.append(val )
		if len(mean)>0:
			arrayMean.append( np.mean(mean))
	return arrayMean

def createMeanArrayTime(matrix,start=0,step=10):
	def calcWMean(row,start,end):
		lastvalue=0
		lasttime=start
		acc=0
		for value, time in row:
			if time < start:
				lastvalue=value
			else:
				if time < end:
					acc=acc+(time-lasttime)*lastvalue
					lasttime=time
					lastvalue=value
				else:
					acc=acc+(end-lasttime)*lastvalue
					break
		return acc/(end-start)
	arrayMean = []
	for interval in range(start,299,step):
		meanW=[]
		for row in matrix:
			meanW.append(calcWMean(row,interval,interval+step) )
		arrayMean.append( np.mean(meanW))
	return arrayMean

def createPrefixArray(arrayMean):
	prefixMean = []
	for i in range(0 , len(arrayMean)):
		if i == 0:
			prefixMean.append(arrayMean[i])
		else:
			prefixMean.append(np.mean([*arrayMean[:i]]))
	return prefixMean

def calculateConfidenceInterval(meanRows):
	m = len(meanRows)
	mSquared = np.sqrt(m)
	meanEstimate = np.mean(meanRows)
	stdDev = np.std(meanRows)
	tValue = t.pdf(0.1, df=m-1)
	confLower = meanEstimate - tValue*(stdDev/mSquared)
	confUpper = meanEstimate + tValue*(stdDev/mSquared)
	return meanEstimate, confLower, confUpper

def calculateExtimatedValue(configuration,moduleName,attributeName,ontime=False):
	print("<--- BEGINNING configuration for attribute %s of module %s --->"%(attributeName,moduleName))
	prefixMean = []

	if ontime:
		configuration[moduleName]["ONTIME"] = createMeanArrayTime(configuration[moduleName][attributeName],step=stepforqueues)
	else:
		configuration[moduleName]["ONTIME"] = createMeanArray(configuration[moduleName][attributeName],step=1)
	prefixMean = createPrefixArray(configuration[moduleName]["ONTIME"])
	trans=calculateTransient(prefixMean)
	if trans >= len(prefixMean):
		count = count + 1
	if trans < 0:
		countneg = countneg + 1
	print("TRANSIENT =",trans)
	if not trans is None:
		deleteNvalues(configuration[moduleName]["ONTIME"],trans)
	configuration[moduleName]["PREFIXMEAN"]=createPrefixArray(configuration[moduleName]["ONTIME"])
	if ontime:
		meanRows = [np.mean(createMeanArrayTime([run],start=trans*stepforqueues,step=stepforqueues)) for run in configuration[moduleName][attributeName] ]
	else:
		meanRows = [np.mean(createMeanArray([run],start=trans*1,step=1)) for run in configuration[moduleName][attributeName] ]
	print(f'MEAN FOR RUN = {meanRows}')
	return (meanRows, trans)

def analyzeList(data):
	print(f'------------------------------------{[]}-------------------------------------------')
	data["MEAN"], data["confLower"], data["confUpper"] = calculateConfidenceInterval(data["list"])
	print(f'ESTIMATED MEAN = {data["MEAN"]}')
	print(f'CONFIDENCE INTERVALS lower: {data["confLower"]}, upper: {data["confUpper"]}')

def printone(f,data,name):
	f.write(str(name)+", ")
	f.write(str(data["trans"])+", ")
	f.write(str(data["confLower"]) +", "+str(data["confUpper"])+",")
	f.write(str(data["MEAN"]))
	f.write("\n")

def creaRisultatiPuliti(total):
	with open(risultatiPuliti,"a") as f:
		f.write("lambda, w, N, K, p, z, name, startTransient, confLower, confUpper, meanValues \n") # titolo
	for a,b,c,d,e,g in iterateOnParams(["lambda", "w", "N", "K", "p", "z"]):
		with open(risultatiPuliti,"a") as f:
			f.write(str(paramsDict["lambda"][a])+", " + str(paramsDict["w"][b])+", " + str(paramsDict["N"][c])+", " +str(paramsDict["K"][d])+", " + str(paramsDict["p"][e])+", " +str(paramsDict["z"][g]+","))
			printone(f,total[a][b][c][d][e][g]["LT"], "lifetime")
			f.write(str(paramsDict["lambda"][a])+", " + str(paramsDict["w"][b])+", " + str(paramsDict["N"][c])+", " +str(paramsDict["K"][d])+", " + str(paramsDict["p"][e])+", " +str(paramsDict["z"][g]+","))
			printone(f,total[a][b][c][d][e][g]["EnergyQL"], "Energy QL")
			f.write(str(paramsDict["lambda"][a])+", " + str(paramsDict["w"][b])+", " + str(paramsDict["N"][c])+", " +str(paramsDict["K"][d])+", " + str(paramsDict["p"][e])+", " +str(paramsDict["z"][g]+","))
			printone(f,total[a][b][c][d][e][g]["CustomerQL"], "Customer QL")
		print("Results saved on file")

def maxinrun(run):
	max=run[0][0]
	for value,time in run:
		if value>max:
			max=value
	return max

def mininrun(run):
	min=run[0][0]
	for value,time in run:
		if value<min:
			min=value
	return min

def main():
	checkArgs()
	total=[]
	for y in range(	len(paramsDict["lambda"])):
		total.append([])
		for z in range(	len(paramsDict["w"])):
			total[y].append([])
			for x in range(	len(paramsDict["N"])):
				total[y][z].append([])
				for k in range(	len(paramsDict["K"])):
					total[y][z][x].append([])
					for s in range(	len(paramsDict["p"])):
						total[y][z][x][k].append([])
						for q in range(	len(paramsDict["z"])):
							total[y][z][x][k][s].append([])
							total[y][z][x][k][s][q]=defaultdict(list)

	deletePrevResults(risultatiSporchi, risultatiPuliti)
	assembleDictionary(total)

	print("-----------Done---------------")
	for inl,inw,inn,ink,inp,inz in iterateOnParams(["lambda","w","N","K","p","z"]):
		configuration = total[inl][inw][inn][ink][inp][inz]

		configuration["LT"] = defaultdict(list)
		configuration["LT"]["list"], configuration["LT"]["trans"] = calculateExtimatedValue(configuration,"sinkC","lifeTime")
		analyzeList(configuration["LT"])

		configuration["MaxLT"] = defaultdict(list)
		configuration["MaxLT"]["list"] = [maxinrun(arr) for arr in configuration["sinkC"]["lifeTime"]]
		analyzeList(configuration["MaxLT"])

		configuration["MinLT"] = defaultdict(list)
		configuration["MinLT"]["list"] = [mininrun(arr) for arr in configuration["sinkC"]["lifeTime"]]
		analyzeList(configuration["MinLT"])

		configuration["CustomerQL"] = defaultdict(list)
		configuration["CustomerQL"]["list"], configuration["CustomerQL"]["trans"] = calculateExtimatedValue(configuration,"customerQueueQ1","queueLength",ontime=True)
		analyzeList(configuration["CustomerQL"])


		configuration["EnergyQL"] = defaultdict(list)
		configuration["EnergyQL"]["list"], configuration["EnergyQL"]["trans"] = calculateExtimatedValue(configuration,"energyQueueQ2","queueLength",ontime=True)
		analyzeList(configuration["EnergyQL"])

	print("----------- Error? --------------------")

	print('count failed transient', count)
	print('count negative transient', countneg)





	creaRisultatiPuliti(total)
	#deletePrevGraph()


	print("----------- Graph --------------------")

	for inn, ink in iterateOnParams(["N","K"]): #capacità
		array=[]
		for i in range(int(300/stepforqueues)):
			array.append([])
		for inp, inl, inw, inz in iterateOnParams(["p","lambda","w", "z"]): #service time
			for time, value in enumerate(total[inl][inw][inn][ink][inp][inz]["customerQueueQ1"]["ONTIME"]):
				array[time].append(value)
		printarray=[]
		for elem in array:
			printarray.append(np.mean(elem))
		printGraphOnTime(printarray, f"CQL ontime for N={paramsDict['N'][inn]} K={paramsDict['K'][ink]}" )
	
	for inn, ink in iterateOnParams(["N","K"]): #capacità
		array=[]
		for i in range(int(300/stepforqueues)):
			array.append([])
		for inp, inl, inw, inz in iterateOnParams(["p","lambda","w", "z"]): #service time
			for time, value in enumerate(total[inl][inw][inn][ink][inp][inz]["customerQueueQ1"]["PREFIXMEAN"]):
				array[time].append(value)
		printarray=[]
		for elem in array:
			printarray.append(np.mean(elem))
		printGraphOnTime(printarray, f"CQL prefixmean for N={paramsDict['N'][inn]} K={paramsDict['K'][ink]}" )

####################################################################################################


	for inn, ink in iterateOnParams(["N","K"]): #capacità
		array=[]
		for i in range(int(300/stepforqueues)):
			array.append([])
		for inp, inl, inw, inz in iterateOnParams(["p","lambda","w", "z"]): #service time
			for time, value in enumerate(total[inl][inw][inn][ink][inp][inz]["energyQueueQ2"]["ONTIME"]):
				array[time].append(value)
		printarray=[]
		for elem in array:
			printarray.append(np.mean(elem))
		printGraphOnTime(printarray, f"EQL ontime for N={paramsDict['N'][inn]} K={paramsDict['K'][ink]}" )
	
	for inn, ink in iterateOnParams(["N","K"]): #capacità
		array=[]
		for i in range(int(300/stepforqueues)):
			array.append([])
		for inp, inl, inw, inz in iterateOnParams(["p","lambda","w", "z"]): #service time
			for time, value in enumerate(total[inl][inw][inn][ink][inp][inz]["energyQueueQ2"]["PREFIXMEAN"]):
				array[time].append(value)
		printarray=[]
		for elem in array:
			printarray.append(np.mean(elem))
		printGraphOnTime(printarray, f"EQL prefixmean for N={paramsDict['N'][inn]} K={paramsDict['K'][ink]}" )

#############################################################################################


	for inn, ink in iterateOnParams(["N","K"]): #capacità
		array=[]
		for i in range(int(300/1)):
			array.append([])
		for inp, inl, inw, inz in iterateOnParams(["p","lambda","w", "z"]): #service time
			for time, value in enumerate(total[inl][inw][inn][ink][inp][inz]["sinkC"]["ONTIME"]):
				array[time].append(value)
		printarray=[]
		for elem in array:
			printarray.append(np.mean(elem))
		printGraphOnTime(printarray, f"LifeTime for N={paramsDict['N'][inn]} K={paramsDict['K'][ink]}" )


	for inn, ink in iterateOnParams(["N","K"]): #capacità
		array=[]
		for i in range(int(300/1)):
			array.append([])
		for inp, inl, inw, inz in iterateOnParams(["p","lambda","w", "z"]): #service time
			for time, value in enumerate(total[inl][inw][inn][ink][inp][inz]["sinkC"]["PREFIXMEAN"]):
				array[time].append(value)
		printarray=[]
		for elem in array:
			printarray.append(np.mean(elem))
		printGraphOnTime(printarray, f"LifeTime prefixmean for N={paramsDict['N'][inn]} K={paramsDict['K'][ink]}" )

	print(max( [ total[inl][inw][inn][ink][inp][inz]["MaxLT"]['MEAN'] for inl,inw,inn,ink,inp,inz in iterateOnParams(["lambda","w","N","K","p","z"]) ] ))
	print(min( [ total[inl][inw][inn][ink][inp][inz]["MinLT"]['MEAN'] for inl,inw,inn,ink,inp,inz in iterateOnParams(["lambda","w","N","K","p","z"]) ] ))
if __name__== "__main__":
	main()
