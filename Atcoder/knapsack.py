import numpy as np

def knapsack(size,value,limit):
    size.insert(0, 0)
    value.insert(0, 0)
    table = np.zeros((len(size),limit+1))
    for i in range(1,len(size)):
        for j in range(1,limit+1):
            if j >= size[i]:
                table[i][j] = max(table[i-1][j] , table[i-1][j-size[i]] + value[i])
            else:
                table[i][j] = table[i-1][j]
    for i in range(len(size)):
        print(table[i,:])

#size =  [2,1,3,2]
#value = [3,2,4,2]

size = [10,4,5,1,7,3,6,3]
value = [7,2,9,4,9,7,4,5]

knapsack(size,value,20)