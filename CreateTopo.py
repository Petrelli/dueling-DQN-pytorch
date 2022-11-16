import sys
import ast

def createTopo(fileName):
	file = open(fileName, "r")
	contents = file.read()
	topo = ast.literal_eval(contents)
	
	file.close()
	return topo

#the following two lines are used for testing
#topology = createTopo(sys.argv[1])
#print(topology)


