def updatePop(contentID, popTable):
    if contentID in popTable:
        popTable[contentID] += 1
    else:
        popTable.update({contentID: 1})
         