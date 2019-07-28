import matplotlib.pyplot as plt

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
    plt.axvline(trans)
    plt.xlabel(descrX)
    plt.ylabel(descrY)
    plt.title(title)
    plt.savefig(graph + title.replace(" ", "_") + "trans" + ".png")
    plt.close()
