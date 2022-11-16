# this function is used to detect the link bandwidth traffic ratio, 
# num of requests, percentage of haptic, popularity, and content type
# of each node. The input is the item of the traffic, e.g., requests[0]
# I'd like to revise the input to the value of requests[0], then the node ID
# which is the key of requests[0] also needs to be passed to the function
#from Evaluation import evaluate
from Popularity import updatePop

def observe(traffic, Nodes, popTable):
    observation = []
    
    for key in traffic.keys():
        nodeID = key
        bandWidth = Nodes[nodeID].bandWidth
        contentSet = []
        load = 0.0
        typeSet = []
        
        for i in range(0, len(traffic[key])):
            contentID = traffic[key][i][0]  # the first item in the vale list is the content ID
            updatePop(contentID, popTable)    # update content popularity
            contentSet.append(contentID)
            if Nodes[nodeID].check(contentID): # reduce the load if the content is cached
                load += 0.0
            else:
                load  += traffic[key][i][1]    # the 2nd ... is the content size
            contentType = traffic[key][i][2]    # the 3rd ... is the requirement aka content type
            typeSet.append(contentType)
        
        BWRatio = load/bandWidth    # link bandwidth load ratio

        numReq = len(traffic[key])  # num of requests
        numHaptic = len([k for k in typeSet if k <= 1])
        perHaptic = numHaptic/numReq  # percentage of haptic content
        
        for conID in contentSet:
            popularity = popTable[conID]
            observation.append([nodeID, conID, BWRatio, numReq, perHaptic, popularity])
        
    return observation


#def observe(traffic, bandWidth, nodeID, popTable):
#    observation = []
    

        
        
        
            