import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from scipy.stats import t
import sys
import math


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
	for j in range(step * 2, len(array), step):
		diff = abs(np.mean(array[0:j]) - np.mean(array[0:j - step]))
		if np.greater(diff, max_diff):
			max_diff = diff
			max_index = j
	return max(max_index, 0)

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
					if  moduleName in moduleList:  #interesting modules
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
											r = rowToStr(row)
											t = rowToStr(rowtime)
											f.write("Configuration, "+nameFile +", Module," +module + ", Attribute," +attribute +  ", Run, "+run+", Values, " + r + ", Time, " + t + "\n")

def createMeanArray(matrix):
	arrayMean = []
	for i in range(0 , minRow(matrix)):
		mean = meanColumn(matrix,i)
		arrayMean.append(mean)
	return arrayMean

def createMeanArrayTime(matrix):
	arrayMean = []
	index=[]
	for i in len(matrix):
		index.append(0)


	flag=True
	#for tindex, tvalue in enumerate(index): tvalue<len(matrix[tindex])
	while Flag:
		i=np.argmin(time for iy, ix in enumerate(index): time=matrix[iy][ix][1] )
		arrayMean.append( np.mean( [state for iy, ix in enumerate(index): state=matrix[iy][ix][0] )) # , matrix[i][index[i]][1]) )
		index[i]=index[i]+1
		#aggiungo un nuovo valore ad arrayMean
		Flag= False
		for tindex, tvalue in enumerate(index):
			Flag= Flag or tvalue<len(matrix[tindex]):

def createPrefixArray(arrayMean):
	prefixMean = []
	for i in range(0 , len(arrayMean)):
		if i == 0:
			prefixMean.append(arrayMean[i])
		else:
			prefixMean.append(np.mean([*arrayMean[:i]]))
	return prefixMean

def calculateMeanRow(matrix):
	meanRows=[]
	for i,row in enumerate(matrix):
		mean = np.mean([x for x,_ in row])
		meanRows.append(mean)
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
	arrayMean = []
	prefixMean = []
	arrayMean = createMeanArray(configuration[moduleName][attributeName])
	prefixMean = createPrefixArray(arrayMean)
	trans=calculateTransient(prefixMean)
	if trans >= len(prefixMean):
		count = count + 1
	if trans < 0:
		countneg = countneg + 1
	print("TRANSIENT =",trans)
	if not trans is None:
		deleteNinitalColumns(configuration[moduleName][attributeName],trans)
		deleteNvalues(arrayMean,trans)
	prefixMean=createPrefixArray(arrayMean)
	meanRows = calculateMeanRow(configuration[moduleName][attributeName])
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
	#LTontime= defaultdict(list)
	#LTontime["list"]=[]
	#LTonmax= defaultdict(list)
	#LTonmax["list"]=[]
	#LTonmin= defaultdict(list)
	#LTonmin["list"]=[]
	print("-----------Done---------------")
	for inl,inw,inn,ink,inp,inz in iterateOnParams(["lambda","w","N","K","p","z"]):
		configuration = total[inl][inw][inn][ink][inp][inz]

		configuration["LT"] = defaultdict(list)
		configuration["LT"]["list"], configuration["LT"]["trans"] = calculateExtimatedValue(configuration,"sinkC","lifeTime")
		analyzeList(configuration["LT"])

		configuration["MaxLT"] = defaultdict(list)
		configuration["MaxLT"]["list"] = [max(arr) for arr in configuration["sinkC"]["lifeTime"]]
		analyzeList(configuration["MaxLT"])

		configuration["MinLT"] = defaultdict(list)
		configuration["MinLT"]["list"] = [min(arr) for arr in configuration["sinkC"]["lifeTime"]]
		analyzeList(configuration["MinLT"])

		configuration["CustomerQL"] = defaultdict(list)
		configuration["CustomerQL"]["list"], configuration["CustomerQL"]["trans"] = calculateExtimatedValue(configuration,"customerQueueQ1","queueLength")
		analyzeList(configuration["CustomerQL"])


		configuration["CustomerQL"]["onTime"] = createMeanArray(configuration["customerQueueQ1"]["queueLength"])

		configuration["EnergyQL"] = defaultdict(list)
		configuration["EnergyQL"]["list"], configuration["EnergyQL"]["trans"] = calculateExtimatedValue(configuration,"energyQueueQ2","queueLength")
		analyzeList(configuration["EnergyQL"])

		#LTontime["list"].append(createMeanArray(configuration["sinkC"]["lifeTime"]))
		#LTonmax["list"].append(max(arr) for arr in configuration["sinkC"]["lifeTime"])

	#finarrLTontime= createMeanArray(LTontime["list"])
	#finarrLTonmax= createMeanArray(LTonmax["list"])
	#finarrLTonmin= createMeanArray(LTonmin["list"])
	#print("finarr",finarr)
	#print("len",len(finarr))

	print("----------- Error? --------------------")

	print('count failed transient', count)
	print('count negative transient', countneg)


	banana = []
	for inl,inw,inn,ink,inp,inz in iterateOnParams(["lambda","w","N","K","p","z"]):
		configuration = total[inl][inw][inn][ink][inp][inz]
		print('len', len(configuration["CustomerQL"]["onTime"]))
		print('LOL', configuration["CustomerQL"]["onTime"])
		banana.append(np.mean(configuration["CustomerQL"]["onTime"]))
	banana = createPrefixArray(banana)
	banana = sorted(banana)
	printGraph(range(0, len(banana)), banana, f"banana", "X", 'Y', 'CustomerQ1 QL')


	#deletePrevGraph()
	print("----------- Graph --------------------")

	#printGraph(range(0, len(finarrLTontime)), finarrLTontime, f"Tempo di permanenza medio nel sistema", "SinkC LifeTime", 'Tempo', 'Lifetime della SinkC')
	#printGraph(range(0, len(finarrLTonmax)), finarrLTonmax, f"Tempo di permanenza max nel sistema", "SinkC LifeTime", 'Tempo', 'Lifetime della SinkC')
	#printGraph(range(0, len(finarrLTonmin)), finarrLTonmin, f"Tempo di permanenza min nel sistema", "SinkC LifeTime", 'Tempo', 'Lifetime della SinkC')

	'''
	capacity = []
	energy = []
	confL = []
	confU = []
	for in1, n1 in enumerate(paramsDict["N"]):
		capacity.append(int(n1))
		energyforconf = []
		conflowerforconf = []
		confupperforconf = []
		for inl, inw, ink, inp, inz in iterateOnParams(["lambda","w","K","p","z"]):
			energyforconf.append(total[inl][inw][in1][ink][inp][inz]["EnergyQL"]["MEAN"])
			conflowerforconf.append(total[inl][inw][in1][ink][inp][inz]["EnergyQL"]["confLower"])
			confupperforconf.append(total[inl][inw][in1][ink][inp][inz]["EnergyQL"]["confUpper"])
		confL.append(np.mean(conflowerforconf))
		confU.append(np.mean(confupperforconf))
		energy.append(np.mean(energyforconf))
	energy = [x for _,x in sorted(zip(capacity, energy))]
	capacity = [x for x,_ in sorted(zip(capacity ,energy))]
	printGraph(capacity, energy,f"Energy queue based on energy queue capacity{capacity}", "Energy QueueLenght", 'N = Queue capacity', 'Energy = Mean energy queue lenght')
	printGraphWConf(capacity, energy, confL, confU, f"Energy queue based on capacity{capacity}", "Energy QueueLenght", 'N = Queue capacity', 'Energy = Mean energy queue lenght')

	for inn, ink in iterateOnParams(["N","K"]): #capacità
		st = []
		job = []
		energy = []
		for inp, inl in iterateOnParams(["p","lambda"]): #service time
			jobforconf = []
			energyforconf = []
			#tmpz = float(paramsDict["z"][inz].replace("s","")) #media
			#tmpp = float(paramsDict["p"][inp].replace("s","")) #fasi
			#print("p=",tmpp)
			#print("z=",val)
			#val = tmpp / tmpz
			#st.append(val)
			#print("val=",val)
			for inw, inz in iterateOnParams(["w", "z"]):
				jobforconf.append(total[inl][inw][inn][ink][inp][inz]["CustomerQL"]["MEAN"])
				energyforconf.append(total[inl][inw][inn][ink][inp][inz]["EnergyQL"]["MEAN"])
			job.append(np.mean(jobforconf))
			energy.append(np.mean(energyforconf))
		job = [x for _,x in sorted(zip(energy, job))]
		energy = [x for x,_ in sorted(zip(energy, job))]
		print("job=",job)
		print("energy=",energy)
		#st = [x for x,_ in sorted(zip(st ,job))]
		printGraph(energy, job, f"Customer queue based on energy queue for K={paramsDict['K'][ink]} and N={paramsDict['N'][inn]}", "Customer QueueLenght", 'EQL', 'CQL')
#def printGraph(x, y, title, label="graph", descrX="x", descrY="y"):
	'''
	for inl, l in enumerate(paramsDict["lambda"]):
		st = []
		job = []
		for inz, z in enumerate(paramsDict["z"]):
			jobforconf = []
			tmpz = float(paramsDict["z"][inz].replace("s",""))
			st.append(tmpz)
			for inn, ink, inw, inp in iterateOnParams(["N","K", "w","p"]):
				jobforconf.append(total[inl][inw][inn][ink][inp][inz]["LT"]["MEAN"])
				job.append(np.mean(jobforconf))
			job = [x for _,x in sorted(zip(st, job))]
			st = [x for x,_ in sorted(zip(st ,job))]
		printGraph(st, job,f"lifetime based on lambda={l}", "Tempo di servizio e di interarrivo", 'Tempo di servizio z', 'Tempo medio di permanenza nel sistema')

	for inl, l in enumerate(paramsDict["lambda"]):
		st = []
		job = []
		for inz, z in enumerate(paramsDict["z"]):
			jobforconf = []
			tmpz = float(paramsDict["z"][inz].replace("s",""))
			st.append(tmpz)
			for inn, ink, inw, inp in iterateOnParams(["N","K", "w","p"]):
				jobforconf.append(total[inl][inw][inn][ink][inp][inz]["MaxLT"]["MEAN"])
				job.append(max(jobforconf))
			job = [x for _,x in sorted(zip(st, job))]
			st = [x for x,_ in sorted(zip(st ,job))]
		printGraph(st, job,f"Max lifetime based on lambda={l}", "Tempo di servizio e di interarrivo", 'Tempo di servizio z', 'Tempo medio di permanenza nel sistema')

	for inl, l in enumerate(paramsDict["lambda"]):
		st = []
		job = []
		for inz, z in enumerate(paramsDict["z"]):
			jobforconf = []
			tmpz = float(paramsDict["z"][inz].replace("s",""))
			st.append(tmpz)
			for inn, ink, inw, inp in iterateOnParams(["N","K", "w","p"]):
				jobforconf.append(total[inl][inw][inn][ink][inp][inz]["MinLT"]["MEAN"])
				job.append(min(jobforconf))
			job = [x for _,x in sorted(zip(st, job))]
			st = [x for x,_ in sorted(zip(st ,job))]
		printGraph(st, job,f"Min lifetime based on lambda={l}", "Tempo di servizio e di interarrivo", 'Tempo di servizio z', 'Tempo medio di permanenza nel sistema')

	for inl, l in enumerate(paramsDict["lambda"]):
		st = []
		job = []
		for inz, z in enumerate(paramsDict["z"]):
			jobforconf = []
			tmpz = float(paramsDict["z"][inz].replace("s",""))
			st.append(tmpz)
			for inn, ink, inw, inp in iterateOnParams(["N","K","w","p"]):
				jobforconf.append(total[inl][inw][inn][ink][inp][inz]["CustomerQL"]["MEAN"])
				job.append(np.mean(jobforconf))
			job = [x for _,x in sorted(zip(st, job))]
			st = [x for x,_ in sorted(zip(st ,job))]
		printGraph(st, job,f"QueueLenght based on lambda={l}", "Tempo di servizio e di interarrivo", 'Tempo di servizio z', 'Lunghezza della coda Q1 dei customer')

	for inw, w in enumerate(paramsDict["w"]):
		st = []
		job = []
		for inz, z in enumerate(paramsDict["z"]):
			jobforconf = []
			tmpz = float(paramsDict["z"][inz].replace("s",""))
			st.append(tmpz)
			for inn, ink, inl, inp in iterateOnParams(["N","K","lambda","p"]):
				jobforconf.append(total[inl][inw][inn][ink][inp][inz]["EnergyQL"]["MEAN"])
				job.append(np.mean(jobforconf))
			job = [x for _,x in sorted(zip(st, job))]
			st = [x for x,_ in sorted(zip(st ,job))]
		printGraph(st, job,f"QueueLenght based on w={w}", "Tempo di servizio e di interarrivo", 'Tempo di servizio z', 'Lunghezza della coda Q2 di energia')


	creaRisultatiPuliti(total)

if __name__== "__main__":
	main()
