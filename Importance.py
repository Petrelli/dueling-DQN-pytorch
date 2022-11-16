# the impTable is used to maintain the importance of each content for each node
# the impTable is a dict which consists of a dict

def updateImp(nodeID, contentID, importance, impTable):
    if nodeID in impTable.keys():
        impTable[nodeID][contentID] = importance
    else:
        tmpDict = {contentID: importance}
        impTable.update({nodeID:tmpDict})
    
