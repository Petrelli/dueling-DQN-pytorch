class Node:
    def __init__(self, maxSize, nodeID, bandWidth): 
        self.ID = nodeID
        self.maxSize = maxSize
        self.cacheSpace = {}
        self.cacheSize = 0.0
        self.bandWidth = bandWidth

    def check(self, content):
        if content in self.cacheSpace.keys():
            return True
        else:
            return False


    def cache(self, importance, content, size):
        # if the node has enough free space, then directly cache the content
        if(self.maxSize - self.cacheSize >= size):
            self.cacheSpace[content] = [importance, size]
            self.cacheSize += size
            # print(f'cache content: {content}, content size is: {size}, current size is: {self.cacheSize}')
        else: 
            # sort the cache space based on the importance of the cached contents
            sortedContent = sorted(self.cacheSpace.items(), key=lambda x:(x[1],x[0])) 
            flag = False
            #for i in range(0, len(self.cacheSpace.keys())): # there is a bug: i should be always the 1st item
            while((self.maxSize - self.cacheSize < size) and (len(sortedContent)>=1)):
                minValue = sortedContent[0][1][0]
                minSize = sortedContent[0][1][1] 
                minContent = sortedContent[0][0]
                # print(f'min content is: {minContent}, its size is: {minSize}, min value is: {minValue}')
                #print(f'minValue is {minValue}, minSize is {minSize}')  just for debugging
                #if last ranked cached content's importance is less than the content's value
                if(minValue <= importance): 
                    # remove the low importance content
                    sortedContent.pop(0)
                    self.cacheSpace.pop(minContent)
                    self.cacheSize -= minSize
                    
                    # if the node now has enough space
                    if(self.maxSize - self.cacheSize >= size):
                        flag = True
                        # print(f'max size is: {self.maxSize}, current size is: {self.cacheSize}')
                        break
                else: 
                    break
               
                
            if(flag):
                self.cacheSpace[content] = [importance, size]
                self.cacheSize += size
            else:
                # do nothing
                nothing = True
                


