
def updateSize(contentID, size, sizeTable):
    if contentID in sizeTable.keys():
        tmpSize = sizeTable[contentID]
        if tmpSize == size:
            # do nothing
            tmpSize = size
        else:
            print("SIZE ERROR!")
    else:
        sizeTable.update({contentID: size})


def updateReq(contentID, requirement, reqTable):
    if contentID in reqTable.keys():
        if requirement == reqTable[contentID]:
            reqTable[contentID] = requirement
        else: 
            print("REQ ERROR!")
    else:
        reqTable.update({contentID: requirement})
         
    