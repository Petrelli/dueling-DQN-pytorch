# author: Zhe Zhang
# the generated traffic format is: [nodeID, [contentID, size, requirement]]
from hashlib import new
import random as rd
import scipy.stats as stats
import sys
import numpy as np
from numpy import random
import matplotlib.pyplot as plt


nodeSet = set([1, 2, 3, 4, 5, 6]) # configurable parameters
contNum = 500 # total number of contents
_4K_30fpsSize = 30 # total size 
_4K_60fpsSize = 60
_8K_30fpsSize = 150
_8K_60fpsSize = 240
_1080P_30fpsSize = 16
_1080P_60fpsSize = 24
hapticSize = 0.1
audioSize = 0.3
meanReq = [150, 200, 150, 250, 350, 300, 500, 700, 600, 300, 250, 150, 200, 220] # the mean value of arrival rate can be configured

conType = [111, hapticSize, audioSize] # 111 stands for video
videoType = [_4K_30fpsSize,_4K_60fpsSize, _8K_30fpsSize, _8K_60fpsSize, _1080P_30fpsSize, _1080P_60fpsSize]

# using a dict to replace switch case statement
latencyReq = {_4K_30fpsSize: 60,
_4K_60fpsSize: 60,
_8K_30fpsSize: 30,
_8K_60fpsSize: 30,
_1080P_30fpsSize: 100,
_1080P_60fpsSize: 100,
hapticSize: 1,
audioSize: 10
}

# generate contents
contents = {}
for i in range(1, contNum + 1):
    cType = rd.sample(conType, 1)[0]
    if(cType == 111):
        tmpType = rd.sample(videoType, 1)[0]
    else:
        tmpType = cType   
    delayReq = latencyReq[tmpType]
    contents.update({i: [tmpType, delayReq]})
        
# generate arrival rate
arrivalRate = []
for i in meanReq:
    for j in range(0, 5): # the 2nd parameter indicates the number of generated values
        # larger 2nd parameter value means a more stable traffic trend
        arrivalRate.append(random.poisson(i, 1)[0]) # poisson distribution

# generate zipf-distribution pmf
prob = [0] * contNum
contentRange = np.arange(1, contNum + 1)
a = 1.1
weights = contentRange ** (-a)
weights /= weights.sum()
data = stats.rv_discrete(name = "bounded_zipf", values = (contentRange, weights))
for i in range(1, contNum + 1):
    prob[i - 1] = data.pmf(i)

# generate traffic that follows zipf-distribution
traffics = []
contentSet = range(1, contNum+1)
for i in arrivalRate:
    req = np.random.choice(contentSet, i, p = prob)
    request = {}
    for j in range(0, len(req)):
        nodeID = rd.sample(nodeSet, 1)[0]
        if nodeID in request.keys():
            oldReq = request[nodeID]
            oldReq.append([req[j], contents[req[j]][0], contents[req[j]][1]])
            request.update({nodeID: oldReq})
        else:
            request.update({nodeID: [[req[j], contents[req[j]][0], contents[req[j]][1]]]})
    traffics.append(request)

# write the generated traffic into a file
total = 0
indexSet = []
numSet = []
for r in range(len(traffics)):
    indexSet.append(r)
    tmpTotal = 0
    for l in traffics[r].keys():
        total += len(traffics[r][l])
        tmpTotal += len(traffics[r][l])
    numSet.append(tmpTotal)
print(total)

fig1 = plt.figure(1)
plt.xlabel("Time (min)")
plt.ylabel("Number of requests")
plt.plot(indexSet, numSet)
plt.show()
fig1.savefig("traffic.pdf")

with open(sys.argv[1], 'w') as f:
    print(traffics, file = f)
    
# the following code is used for testing 
#print(traffics)
