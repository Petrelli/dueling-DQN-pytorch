from RoutingSimple import calculate_distances
def evaluate(nodeID, contentID, nodes, requirement, rTable):
    hit = 0
    latency = 0
    unSatNum = 0
    server = True
    neighbor = 999
    if nodes[nodeID].check(contentID):
        hit = 1
        latency = 1
        server = False
    else:
        minCost = 100000
        neighbor = contentID
        flag = False
        for k in nodes.keys():
            if(nodeID != k and nodes[k].check(contentID)):
                cost = calculate_distances(rTable, nodeID, k)
                hit = 1
                server = False
                flag = True
                #print(f'cost is: {cost}, min cost is: {minCost}')
                if minCost > cost:
                    minCost = cost
                    
                    neighbor = k
        if flag:
            latency = minCost
        
        if requirement < latency:
            unSatNum = 1
       
    
    if server:
        latency = 10
        
    return hit, latency, unSatNum, neighbor
        
def getPopRatio(contentID, traffic):
    
    load = len(traffic)
    popularity = 0.0
    for i in range(len(traffic)):
        if traffic[i][0] == contentID:
            popularity += 1
    
    popRatio = popularity/load
    
    return popRatio
      
def detectCongestion(contentID, sizeTable, linkBW, currentTraffic):
    size = sizeTable[contentID]
    currentTraffic += size
    congestion = False
    if linkBW <= currentTraffic:
        congestion = True
    return congestion    
    
        
            
    