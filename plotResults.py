import matplotlib.pyplot as plt
import numpy as np

graph = "../graph/"

def printGraphOnTime(y, title, descrX="x", descrY="y"):
	plt.plot(y)
	plt.xlabel(descrX)
	plt.ylabel(descrY)
	plt.title(title)
	plt.savefig(graph + title.replace(" ", "_") + ".png")
	plt.close()

def printMultiGraph(multi, title, descrX="x", descrY="y"):
	labels = []
	ss = []
	for y,s in multi:
		tmp, = plt.plot(y, label=s)
		labels.append(tmp)
		ss.append(s)
	plt.xlabel(descrX)
	plt.ylabel(descrY)
	plt.title(title)
	plt.legend(labels, ss)
	plt.savefig(graph + title.replace(" ", "_") + ".png")
	plt.close()

def printGraphTransient(y, trans, title, descrX="x", descrY="y"):
	plt.plot(y)
	plt.axvline(trans, color="red")
	plt.xlabel(descrX)
	plt.ylabel(descrY)
	if len(y) < 100:
		plt.xticks(np.arange(0, len(y), 5))
	else:
		plt.xticks(np.arange(0, len(y), 20)) 
	plt.title(title)
	plt.savefig(graph + title.replace(" ", "_") + "_trans" + ".png")
	plt.close()
