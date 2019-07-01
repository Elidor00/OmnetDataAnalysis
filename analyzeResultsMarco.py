import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from scipy.stats import t
import sys
import math

risultatiPuliti= "../risultatiPuliti.csv"
risultatiSporchi= "../risultatiSporchi.csv"
graphs="./graphs/"

attrDict={
"passiveQueue":["queueLength","dropped"],
"passiveQueue1":["queueLength","dropped"],
"sink":["lifeTime"],
"sink1":["lifeTime"] }

loadbalanceparamsDict={
  "y":["0.5s", "0.6s", "0.7s"],
  "x":["0.25s", "0.30s"],
  "z":["0.3s", "0.5s", "0.8s"],
  "K":["7","8","9","10"],
  "s":["random", "roundRobin", "shortestQueue"]
}

distributedloadbalanceparamsDict={
  "y":["0.05s", "0.06s", "0.07s"],
  "x":["0.1s", "0.15s", "0.20s"],
  "z":["0.02s", "0.03s", "0.04s"],
  "K":["18","19","20","21"],
  "d":["3","4","5"]
}

#paramsDict=distributedloadbalanceparamsDict
paramsDict=loadbalanceparamsDict

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
	plt.xlabel(descrX)
	plt.ylabel(descrY)
	plt.title(title)
	plt.savefig(graphs+title.replace(" ","_")+".png")
	#plt.show()
	plt.close()

def printMultiGraph(multi,title,descrX="x",descrY="y"):
	labels=[]
	ss=[]
	for x,y,s in multi:
		plt.plot(x,y, 'ro')
		tmp,=plt.plot(x,y, label=s)
		labels.append(tmp)
		ss.append(s)
	plt.xlabel(descrX)
	plt.ylabel(descrY)
	plt.title(title)
	plt.legend(labels,ss)
	plt.savefig(graphs+title.replace(" ","_")+".png")
	#plt.show()
	plt.close()


def calculateTransient(array):
	step = math.ceil(np.multiply(np.true_divide(len(array), 100), 2))
	max_diff = 0
	max_index = -1
	for j in range(step * 2, len(array), step):
		diff = abs(np.mean(array[0:j]) - np.mean(array[0:j - step]))
		if np.greater(diff, max_diff):
			max_diff = diff
			max_index = j
	return max_index

def deleteNinitalColumns(matrix, value):
	for _ in range(0, value):
		for row in matrix:
			del row[0]

def deleteNvalues(array,value):
	for i in range(0,value):
		del array[0]

def meanColumn(matrix, i):
	sum = 0
	numrows = len(matrix)
	for row in matrix:
		sum = sum + row[i][0]
	return sum / float(numrows)

def minRow(matrix): #rows has different lenght
	numrows = len(matrix)
	min = len(matrix[0])
	for i in range(1, numrows):
		if len(matrix[i]) < min:
			min = len(matrix[i])
	return min

def rowToStr(row):
	str1 = ''.join(str(e) for e in row)
	return(str1.replace("]","").replace("[","").replace(",",""))

def checkArgs():
	if (len(sys.argv)!=2):
		print("Insert name of project to analyze")
		exit(0)

def deletePrevResults(): #aggiungere risultatiPuliti come primo argomento
	try:
		os.remove(risultatiPuliti)
	except OSError:
		pass
	try:
		os.remove(risultatiSporchi)
	except OSError:
		pass
	try:
		for g in os.listdir(graphs):
			print
			os.remove(os.path.join(graphs, g))
	except OSError as e:
		print(str(e))
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
					if  moduleName in moduleList:  #interesting modules
						print("MODULE:",str(moduleName))
						for attribute in attributes:
							if (str(attribute)!= "nan") :
								attributeName=str(attribute).split(":")[0]
								if (attributeName in attrDict[moduleName]):
									rowvalue= csv[(csv.type=='vector') & (csv.module==module) & (csv.name==attribute)].vecvalue.describe()
									rowtime= csv[(csv.type=='vector') & (csv.module==module) & (csv.name==attribute)].vectime.describe()
									if len(rowvalue.values) > 2:
										print("attr=",attribute)
										lll=[(float(a),float(b)) for a,b in zip(rowvalue[2].split(),rowtime[2].split() )]
										#print("lll",str(lll))
										lll.insert(0,(0,0))#initialize value of array
										actual[moduleName][attributeName].append(lll)
										with open(risultatiSporchi,"a") as f:
											r=rowToStr(rowvalue[2])
											t=rowToStr(rowtime[2])
											f.write("Configuration, "+nameFile +", Module," +module + ", Attribute," +attribute + ", Run, "+run+", Values, " + r + ", Time, " + t + "\n")
											#f.write("Configuration, "+nameFile +", Module," +module + ", Attribute," +attribute +  ", Run, "+run+", Values, " + r + "\n")

def createMeanArray(matrix):
	arrayMean=[]
	for i in range(0 , minRow(matrix)):
		mean=meanColumn(matrix,i)
		arrayMean.append(mean)
	return arrayMean

def createMeanArrayTime(matrix):

	def checkEnd(index):
		flag= False
		for tindex, tvalue in enumerate(index):
			flag= flag or tvalue<len(matrix[tindex])
		return flag

	def calcMin(index):
		minVal=301
		for indexrow, indexcolumn in enumerate(index):
			if indexcolumn < len(matrix[indexrow]):
				if matrix[indexrow][indexcolumn][1] < minval:
					minval = matrix[indexrow][indexcolumn][1]
		mins=[]
		for indexrow, indexcolumn in enumerate(index):
			if matrix[indexrow][indexcolumn][1] == minval:
				mins.append(indexrow)
		return mins

	def calcMean(index):
		np.mean( [ matrix[indexrow][indexcolumn-1][0] for indexrow, indexcolumn in enumerate(index) ] )

	arrayMean = []
	index=[]
	for i in range(len(matrix)):
		index.append(0)

	while checkEnd(index):
		mins=calcMin(index)# np.argmin([ indexcolumn<len(matrix[index]) matrix[indexrow][indexcolumn][1] for indexrow, indexcolumn in enumerate(index) ])
		for i in mins:
			index[i]=index[i]+1
		arrayMean.append( calcMean()) # , matrix[i][index[i]][1]) )
	return arrayMean


def createPrefixArray(arrayMean):
	prefixMean = []
	for i in range(0 , len(arrayMean)):
		if i == 0:
			prefixMean.append(arrayMean[i])
		else:
			prefixMean.append(np.mean([*arrayMean[:i]]))
	return prefixMean

def calculateMeanRow(matrix):
	#pesare media
	def mediaPesata(row):
		c=0
		time=0
		for val, t in row:
			c=c+val*(t-time)
			time=t
		return (c/len(row))
	meanRows=[]
	for row in matrix:
		meanRows.append(mediaPesata(row))
	return(meanRows)

def calculateConfidenceInterval(meanRows):
	m = len(meanRows)
	mSquared = np.sqrt(m)
	meanEstimate = np.mean(meanRows)
	stdDev = np.std(meanRows)
	tValue = t.pdf(0.1, df=m-1)
	confLower = meanEstimate - tValue*(stdDev/mSquared)
	confUpper = meanEstimate + tValue*(stdDev/mSquared)
	return meanEstimate, confLower, confUpper

def calculateExtimatedValue(configuration,moduleName,attributeName):
	print("<--- BEGINNING configuration for attribute %s of module %s --->"%(attributeName,moduleName))
	arrayMean= []
	prefixMean = []
	arrayMean = createMeanArray(configuration[moduleName][attributeName])
	prefixMean=createPrefixArray(arrayMean)
	trans=calculateTransient(prefixMean)
	print("TRANSIENT=",trans)
	if not trans is None:
		deleteNinitalColumns(configuration[moduleName][attributeName],trans)
		deleteNvalues(arrayMean,trans)
	prefixMean=createPrefixArray(arrayMean)
	meanRows = calculateMeanRow(configuration[moduleName][attributeName])
#	print(f'MEAN FOR RUN = {meanRows}')
	return meanRows,trans

def analyzeList(data): #lista di liste
	data["MEAN"], data["confLower"], data["confUpper"] = calculateConfidenceInterval(data["list"])
#	print(f'TOTAL MEAN = {data["MEAN"]}')
#	print(f'CONFIDENCE INTERVALS lower: {data["confLower"]}, upper: {data["confUpper"]}')

def printone(f,data,name):
	f.write(str(name)+", ")
	f.write(str(data["trans"])+", ")
	f.write(str(data["confLower"]) +", "+str(data["confUpper"])+",")
	f.write(str(data["MEAN"]))
	f.write("\n")

def creaRisultatiPuliti(total):
	with open(risultatiPuliti,"a") as f:
		f.write(  "y, z, x, K, s, name, startTransient, confLower, confUpper, meanValues \n")
	for y,z,x,k,s in iterateOnParams(["y", "z", "x", "K", "s"]):

		with open(risultatiPuliti,"a") as f:
			f.write(str(paramsDict["y"][y])+", " + str(paramsDict["z"][z])+", " + str(paramsDict["x"][x])+", " +str(paramsDict["K"][k])+", " + str(paramsDict["s"][s])+",")
			printone(f,total[y][z][x][k][s]["W"], "mean service time")
			f.write(str(paramsDict["y"][y])+", " + str(paramsDict["z"][z])+", " + str(paramsDict["x"][x])+", " +str(paramsDict["K"][k])+", " + str(paramsDict["s"][s])+",")
			printone(f,total[y][z][x][k][s]["rho"], "mean occupation")
			f.write(str(paramsDict["y"][y])+", " + str(paramsDict["z"][z])+", " + str(paramsDict["x"][x])+", " +str(paramsDict["K"][k])+", " + str(paramsDict["s"][s])+",")
			printone(f,total[y][z][x][k][s]["DropRate"], "mean DropRate")
			f.write(str(paramsDict["y"][y])+", " + str(paramsDict["z"][z])+", " + str(paramsDict["x"][x])+", " +str(paramsDict["K"][k])+", " + str(paramsDict["s"][s])+",")
			printone(f,total[y][z][x][k][s]["Throughput"], "mean Throughput")
		print("Results saved on file")

def main():
	checkArgs()
	total=[]
	for y in range(	len(paramsDict["y"])):
		total.append([])
		for z in range(	len(paramsDict["z"])):
			total[y].append([])
			for x in range(	len(paramsDict["x"])):
				total[y][z].append([])
				for k in range(	len(paramsDict["K"])):
					total[y][z][x].append([])
					for s in range(	len(paramsDict["s"])):
						total[y][z][x][k].append([])
						total[y][z][x][k][s]=defaultdict(list)

	deletePrevResults()
	assembleDictionary(total)
	print("-----------Done---------------")
	for iny,inz,inx,ink,ins in iterateOnParams(["y","z","x","K","s"]):
	#	print("BEGINNING configuration y=%s x=%s z=%s k=%s s=%s"%(y,x,z,k,s))
		configuration=total[iny][inz][inx][ink][ins]
		configuration["W"]=defaultdict(list)
		tmp1,tr1=calculateExtimatedValue(configuration,"sink","lifeTime")
		tmp2,tr2=calculateExtimatedValue(configuration,"sink1","lifeTime")
		configuration["W"]["trans"]= (tr1+tr2)/2
		configuration["W"]["list"]=np.array([tmp1,tmp2] )
		analyzeList(configuration["W"])

		configuration["rho"]=defaultdict(list)
		tmp1,tr=calculateExtimatedValue(configuration,"passiveQueue","queueLength")
		tmp2,tr1=calculateExtimatedValue(configuration,"passiveQueue1","queueLength")
		configuration["rho"]["trans"]= (tr+tr1)/2
		configuration["rho"]["list"]= [r1+r2 for r1,r2 in zip(
			tmp1, tmp2)
		]
		analyzeList(configuration["rho"])

		configuration["DropRate"]=defaultdict(list)
		drop0=[len(dropped)  for dropped in configuration["passiveQueue"]["dropped"]   ]
		drop1=[len(dropped)  for dropped in configuration["passiveQueue1"]["dropped"]   ]
		configuration["DropRate"]["list"]= [r1+r2 for r1,r2 in zip(drop0,drop1) ]
		analyzeList(configuration["DropRate"])

		configuration["Throughput"]=defaultdict(list)
		T0=[len(numberUser)  for numberUser in configuration["sink"]["lifeTime"]   ]
		T1=[len(numberUser)  for numberUser in configuration["sink1"]["lifeTime"]   ]
		configuration["Throughput"]["list"]= [(r1+r2)/300 for r1,r2 in zip(T0,T1) ]
		analyzeList(configuration["Throughput"])


	multi=[]
	for is1, s1 in enumerate(paramsDict["s"]):
		W=[]
		yz=[]
		for inz,iny in iterateOnParams(["z","y"]):
			yz.append( 1/(1/float(paramsDict["z"][inz].replace("s",""))+1/float(paramsDict["y"][iny].replace("s",""))) )
			Wforconf=[]
			for inK,inx in iterateOnParams(["K","x"]):
				Wforconf.append(total[iny][inz][inx][inK][is1]["W"]["MEAN"])
			W.append(np.mean(Wforconf))
		W = [x for _,x in sorted(zip(yz,W))]
		yz = [x for x,_ in sorted(zip(yz,W))]
		multi.append([yz,W,s1])
	printMultiGraph(multi,"mean time spent in the system for servicetime")

	multi=[]
	for is1, s1 in enumerate(paramsDict["s"]):
		W=[]
		yz=[]
		for inz,iny in iterateOnParams(["z","y"]):
			yz.append( 1/(1/float(paramsDict["z"][inz].replace("s",""))+1/float(paramsDict["y"][iny].replace("s",""))) )
			Wforconf=[]
			for inK,inx in iterateOnParams(["K","x"]):
				Wforconf.append(total[iny][inz][inx][inK][is1]["DropRate"]["MEAN"])
			W.append(np.mean(Wforconf))
		W = [x for _,x in sorted(zip(yz,W))]
		yz = [x for x,_ in sorted(zip(yz,W))]
		multi.append([yz,W,s1])
	printMultiGraph(multi,"droprate for servicetime")

	multi=[]
	for is1, s1 in enumerate(paramsDict["s"]):
		W=[]
		yz=[]
		for inz,iny in iterateOnParams(["z","y"]):
			yz.append( 1/(1/float(paramsDict["z"][inz].replace("s",""))+1/float(paramsDict["y"][iny].replace("s",""))) )
			Wforconf=[]
			for inK,inx in iterateOnParams(["K","x"]):
				Wforconf.append(total[iny][inz][inx][inK][is1]["rho"]["MEAN"])
			W.append(np.mean(Wforconf))
		W = [x for _,x in sorted(zip(yz,W))]
		yz = [x for x,_ in sorted(zip(yz,W))]
		multi.append([yz,W,s1])
	printMultiGraph(multi,"rho for servicetime")

	multi=[]
	for is1, s1 in enumerate(paramsDict["s"]):
		W=[]
		yz=[]
		for inz,iny in iterateOnParams(["z","y"]):
			yz.append( 1/(1/float(paramsDict["z"][inz].replace("s",""))+1/float(paramsDict["y"][iny].replace("s",""))) )
			Wforconf=[]
			for inK,inx in iterateOnParams(["K","x"]):
				Wforconf.append(total[iny][inz][inx][inK][is1]["Throughput"]["MEAN"])
			W.append(np.mean(Wforconf))
		W = [x for _,x in sorted(zip(yz,W))]
		yz = [x for x,_ in sorted(zip(yz,W))]
		multi.append([yz,W,s1])
	printMultiGraph(multi,"Throughput for servicetime")

	for inx, x1 in enumerate(paramsDict["x"]):
		multi=[]
		for is1, s1 in enumerate(paramsDict["s"]):
			W=[]
			yz=[]
			for inz,iny in iterateOnParams(["z","y"]):
				yz.append( 1/(1/float(paramsDict["z"][inz].replace("s",""))+1/float(paramsDict["y"][iny].replace("s",""))) )
				Wforconf=[]
				for inK, _ in enumerate(paramsDict["K"]):
					Wforconf.append(total[iny][inz][inx][inK][is1]["DropRate"]["MEAN"])
				W.append(np.mean(Wforconf))
			W = [x for _,x in sorted(zip(yz,W))]
			yz = [x for x,_ in sorted(zip(yz,W))]
			multi.append([yz,W,s1])
		printMultiGraph(multi,"droprate for servicetime for x="+str(x1))



	creaRisultatiPuliti(total)

if __name__== "__main__":
	main()
