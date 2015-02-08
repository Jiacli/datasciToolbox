# script to evaluate the clusters
import sys

if len(sys.argv) != 3:
	print "Expect an input document clustering results file, gold standard file"
	print "Please give the files in that order."
	print "For example, python eval.py docCluster.txt goldClusters.txt"
	sys.exit()

input = open(sys.argv[1], "r")
gold = open(sys.argv[2], "r")

sysClusters = []
goldClusters = []

for line in input:
	tokens = line.strip().split(" ")
	cluster = int(tokens[1])
	doc = int(tokens[0])
	while len(sysClusters) <= cluster:
		sysClusters.append([])
	sysClusters[cluster].append(doc)
input.close()

goldDict = dict()
eventCounter = 0
docCounter = 0
for line in gold:
	cluster = line.strip()

	if cluster != "unlabeled":
		if cluster not in goldDict:
			goldDict[cluster] = eventCounter
			eventCounter += 1

		clusterID = goldDict[cluster]
		while len(goldClusters) <= clusterID:
			goldClusters.append([])
		goldClusters[clusterID].append(docCounter)

	docCounter += 1
gold.close()


# for each gold cluster, find the system cluster that maximizes F1
clusterF1s = []
for goldCluster in goldClusters:
	bestF1 = -1

	for sysCluster in sysClusters:
		tp = 0
		fp = 0
		fn = 0
		for item in goldCluster:
			if item in sysCluster:
				tp += 1.0
			else:
				fn += 1.0
		for item in sysCluster:
			if item not in goldCluster:
				fp += 1.0

		# if none match, just ignore	
		if tp == 0:
			continue

		precision = tp / (tp+fp)
		recall = tp / (tp+fn)
		f1 = 2*precision*recall/(precision+recall)

		if f1 > bestF1:
			bestF1 = f1
	
	clusterF1s.append(bestF1)

macroF1 = 0
for item in clusterF1s:
	macroF1 += item
macroF1 = macroF1 / len(clusterF1s)

print "Macro F1 = " + str(macroF1)





