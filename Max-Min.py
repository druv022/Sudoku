import numpy

def maxMin():
    f =open("output-Data0.csv")
    a = 1
    b = f.readlines()
    c = []
    maxval =0
    minval = 0
    tempMax = 0
    tempMin = 0
    for i in range(len(b)):
        c = b[i].split(',')
        if type(c[0]) == str:
            continue
        tempMax = c[a]
        tempMin = c[a]
        if maxval < tempMax:
            maxval = tempMax
        if minval > tempMin:
            minval = tempMin

    return maxval,minval
