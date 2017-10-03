import json
import pycosat
import subprocess
from sys import argv

import matplotlib.pyplot as plt
import numpy as np

M = 9
problemset = 1000  # to be replaced with 100


def load():
    quizzes = np.zeros((1000000, 81), np.int32)
    solutions = np.zeros((1000000, 81), np.int32)
    for i, line in enumerate(open('sudoku.csv', 'r').read().splitlines()[1:]):
        quiz, solution = line.split(",")
        for j, q_s in enumerate(zip(quiz, solution)):
            q, s = q_s
            quizzes[i, j] = q
            solutions[i, j] = s
    quizzes = quizzes.reshape((-1, 9, 9))
    solutions = solutions.reshape((-1, 9, 9))
    return quizzes, solutions


def countGivens(quizzes):
    b = np.array([0])
    i = -1
    while i < 100001:
        i += 1
        if np.count_nonzero(quizzes[i]) in b:
            continue
        b = np.append(b, [np.count_nonzero(quizzes[i])])

    return b


def changeGiven(matrix, givens):
    c, d = np.nonzero(matrix)  # c= row, d= column
    e = np.count_nonzero(matrix)
    f = np.shape(c)
    i = 0
    testMatrix = np.copy(matrix)
    while i < (e - givens):
        j = np.random.randint(0, f[0])
        testMatrix[c[j], d[j]] = 0
        c, d = np.nonzero(testMatrix)
        f = np.shape(c)
        i += 1
    return testMatrix


def changeMatrix(matrix1, matrix2):
    i = 0
    n = 0
    newQuizMatrix = np.zeros((100000, 81), np.int32).reshape((-1, 9, 9))
    # newQuizMatrix = newQuizMatrix.reshape((-1,9,9))
    while i < 81:
        j = 0
        while j < 100:  # change here for 1000
            a = np.random.randint(0, 100000)
            if (np.count_nonzero(matrix1[a]) < i):
                newQuizMatrix[n] = changeGiven(matrix2[a], i)
            else:
                newQuizMatrix[n] = changeGiven(matrix1[a], i)
            j += 1
            n += 1

        i += 1

    return newQuizMatrix


def changesymGiven(matrix, givens):
    c, d = np.nonzero(matrix)  # c= row, d= column
    e = np.count_nonzero(matrix)
    f = np.shape(c)
    i = 0
    while i < (e - givens):
        j = np.random.randint(0, f[0])
        matrix[c[j], d[j]] = 0
        c, d = np.nonzero(matrix)
        f = np.shape(c)
        i += 1
    return matrix


def changesymMatrix(matrix1, matrix2):
    i = 0
    n = 81
    symQuizMatrix = np.zeros((100000, 81), np.int32).reshape((-1, 9, 9))
    while i < 100:
        a = np.random.randint(0, 100000)
        testMatrix = np.copy(matrix2[a])
        j = 81
        while j > -1:
            symQuizMatrix[n] = changesymGiven(testMatrix, j)
            j += -1
            n += -1
        n = (i*100)+81
        i += 1
    return symQuizMatrix


def sudokuProblem(matrix):
    row, col = np.nonzero(matrix)
    prob = []
    for i in range(row.shape[0]):
        prob.append(
            [100 * np.asscalar((row[i] + 1)) + 10 * (np.asscalar(col[i] + 1)) + np.asscalar(matrix[row[i]][col[i]])])
    return prob


def get_rowrules(M):
    """
    get_rowrules returns the rules that make sure no duplicates can
    be in one row.
    """
    f = open("result.txt", 'w')
    rangem = range(1, M + 1, 1)
    rowrule = []
    x = ""
    y = " row rules"
    for row in rangem:
        for val in rangem:
            temp1 = []
            y += "\n"
            for i in range(1, M + 1, 1):
                temp1.append((row * 100) + (i * 10) + val, )
                y += str((row * 100) + (i * 10) + val) + "\t"
            if len(temp1) != 0:
                rowrule.append(temp1)
            for col in rangem:
                for j in range(col, M + 1, 1):
                    temp = []
                    if col == j:
                        continue
                    temp.append(-((row * 100) + (col * 10) + (val)))
                    temp.append(-((row * 100) + (j * 10) + (val)))
                    x += "-" + str((row * 100) + (col * 10) + (val)) + " \t : " + "-" + str(
                        (row * 100) + (j * 10) + (val)) + "\n"
                    rowrule.append(temp)
    f.write(y)
    f.write(x)
    f.close()
    return rowrule


def get_columnrules(M):
    """
    get_columnrules returns the rules that make sure no duplicated can
    be in a column.
    """
    f = open("result.txt", 'a')
    x = ""
    y = " col rules"
    rangem = range(1, M + 1, 1)
    colrule = []
    for col in rangem:
        for val in rangem:
            temp1 = []
            y += "\n"
            for i in range(1, M + 1, 1):
                temp1.append((i * 100) + (col * 10) + val, )
                y += str((i * 100) + (col * 10) + val) + "\t"
            if len(temp1) != 0:
                colrule.append(temp1)
            for row in rangem:
                for j in range(row, M + 1, 1):
                    temp = []
                    if row == j:
                        continue
                    temp.append(-((row * 100) + (col * 10) + (val)))
                    temp.append(-((j * 100) + (col * 10) + (val)))
                    colrule.append(temp)
                    x += "-" + str((row * 100) + (col * 10) + (val)) + " \t : " + "-" + str(
                        (j * 100) + (col * 10) + (val)) + "\n"

    f.write(y)
    f.write(x)
    f.close()
    return colrule


def get_boxrules(M):
    """
    get_boxrules returns the rules that make sure there can be no
    duplicates in one box.
    """
    N = M
    sq = 3
    a = int(N / sq)
    boxrules = []
    rangeM = range(1, M + 1)
    f = open("result.txt", 'a')
    x = ""
    y = " box rules"
    num1 = 1
    for s in range(1, a + 1):  # 1 to 3
        for t in range(1, a + 1):  # 1 to 3
            for r in range((s * (a)) - 2, ((s + 1) * (a)) - 2):  # 1-4:4-7:7-10
                for c in range((t * (a)) - 2, ((t + 1) * (a)) - 2):
                    temp2 = []
                    if num1 == 10:
                        num1 = 1
                    for r2 in range((s * (a)) - 2, ((s + 1) * (a)) - 2):
                        for c2 in range((t * (a)) - 2, ((t + 1) * (a)) - 2):
                            # comment this out to see if your python doesn't suck.
                            # it doesn't error iff it sucks
                            for i in range(1, N + 1):
                                temp = []
                                if r != r2 and c != c2:
                                    temp.append(-((r * 100) + (c * 10) + (i)))
                                    temp.append(-((r2 * 100) + (c2 * 10) + (i)))
                                if len(temp) == 0:
                                    continue
                                x += "-" + str((r * 100) + (c * 10) + (i)) + " \t : " + "-" + str(
                                    (r2 * 100) + (c2 * 10) + (i)) + "\n"

                                boxrules.append(temp)

                            num = ((r2 * 100) + (c2 * 10) + (num1))

                            if num in temp2:
                                continue
                            temp2.append(num)
                            y += str(num) + "\t"
                    y += "\n"
                    boxrules.append(temp2)
                    temp2 = []
                    num1 += 1
    f.write(y)
    f.write(x)
    f.close()
    return boxrules


def at_least_one_rules(M):
    """
    at_least_one_rule returns the rules that make sure there has
    to be at least one number in a square.
    """
    f = open("result.txt", 'a')
    y = " at least one rule"
    rangem = range(1, M + 1, 1)
    atleastrule = []
    for row in rangem:
        for col in rangem:
            temp = []
            for val in rangem:
                temp.append((row * 100) + (col * 10) + val, )
                y += str((row * 100) + (col * 10) + val) + "\t"
            atleastrule.append(temp)
            y += "\n"
    f.write(y)
    f.close()
    return atleastrule


def get_rules(M):
    """
    get_rules returns all the suduko rules of a sudo of size M.
    """
    cnf1 = get_rowrules(M)
    cnf2 = get_columnrules(M)
    cnf3 = get_boxrules(M)
    cnf4 = at_least_one_rules(M)

    return cnf1 + cnf2 + cnf3 + cnf4


quizzes, solutions = load()
print("Load complete")

b = countGivens(quizzes)
newMatrix = np.zeros((100000, 81), np.int32)
newMatrix = changeMatrix(quizzes, solutions)

symMatrix = np.zeros((100000, 81), np.int32)
symMatrix = changesymMatrix(quizzes, solutions)

prob = sudokuProblem(newMatrix[1767])


def solveSingle(sudoku, verbose):
    prob = sudokuProblem(sudoku)
    a = pycosat.solve(get_rules(9) + prob, 5)
    sol1 = [x for x in a if x > 0]
    return sol1


def displayMatrix(matrix):
    solMatrix = np.zeros((9, 9))
    for i in range(len(matrix)):
        row = int(matrix[i] / 100)
        col = int((matrix[i] - (row * 100)) / 10)
        val = matrix[i] - (row * 100) - (col * 10)
        solMatrix[row - 1][col - 1] = val
    return solMatrix


def getProblem(number, symmetry=0,type2=0):
    text = ""
    if symmetry == 0 and type2 == 0:
        npmatrix = newMatrix[number]
        prob = sudokuProblem(npmatrix)
        text += "None" + "\n"
        return prob, text
    else:
        npmatrix = np.copy(symMatrix[number])
    if int(symmetry / 8) == 1:
        c = checkevenandequal(npmatrix)
        text += "Element switch;"
        npmatrix = switchelement(c, npmatrix)
    symmetry = symmetry % 8
    if int(symmetry / 4) == 1:
        text += "Diagonal switch;"
        npmatrix = switchdiagonally(npmatrix)
    symmetry = symmetry % 4
    if int(symmetry / 2) == 1:
        text += "Column switch;"
        npmatrix = switchcolumnstrips(npmatrix)
    symmetry = symmetry % 2
    if symmetry == 1:
        text += "row switch;"
        npmatrix = switchrowstrips(npmatrix)
    prob = sudokuProblem(npmatrix)
    return prob, text


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


first = 'second'
last = '[-1'

# change here for 1000
onegivenVariables = np.zeros(100)
onegivenOriginal = np.zeros(100)
onegivenLeanrned = np.zeros(100)
onegivenAgility = np.zeros(100)
onegivenLevel = np.zeros(100)
onegivenUsed = np.zeros(100)
onegivenSeconds = np.zeros(100)
onegivenConflicts = np.zeros(100)
onegivenLimit = np.zeros(100)
onegivenMB = np.zeros(100)

symgivenVariables = np.zeros((100, 81))
symgivenOriginal = np.zeros((100, 81))
symgivenLeanrned = np.zeros((100, 81))
symgivenAgility = np.zeros((100, 81))
symgivenLevel = np.zeros((100, 81))
symgivenUsed = np.zeros((100, 81))
symgivenSeconds = np.zeros((100, 81))
symgivenConflicts = np.zeros((100, 81))
symgivenLimit = np.zeros((100, 81))
symgivenMB = np.zeros((100, 81))

avggivenVariables = np.zeros(81)
avggivenOriginal = np.zeros(81)
avggivenLeanrned = np.zeros(81)
avggivenAgility = np.zeros(81)
avggivenLevel = np.zeros(81)
avggivenUsed = np.zeros(81)
avggivenSeconds = np.zeros(81)
avggivenConflicts = np.zeros(81)
avggivenLimit = np.zeros(81)
avggivenMB = np.zeros(81)

maxgivenVariables = np.zeros(81)
maxgivenOriginal = np.zeros(81)
maxgivenLeanrned = np.zeros(81)
maxgivenAgility = np.zeros(81)
maxgivenLevel = np.zeros(81)
maxgivenUsed = np.zeros(81)
maxgivenSeconds = np.zeros(81)
maxgivenConflicts = np.zeros(81)
maxgivenLimit = np.zeros(81)
maxgivenMB = np.zeros(81)

mingivenVariables = np.zeros(81)
mingivenOriginal = np.zeros(81)
mingivenLeanrned = np.zeros(81)
mingivenAgility = np.zeros(81)
mingivenLevel = np.zeros(81)
mingivenUsed = np.zeros(81)
mingivenSeconds = np.zeros(81)
mingivenConflicts = np.zeros(81)
mingivenLimit = np.zeros(81)
mingivenMB = np.zeros(81)


def splitKeyValue(splitdata):
    # splits data
    splita = splitdata[0].split()
    splitb = splitdata[1].split()
    splitc = splitdata[3].split()
    splitd = splitdata[4].split()
    splite = splitdata[8].split()
    keyvalpair = []
    num1 = 0
    num2 = 0
    for i in range(len(splite)):
        temp = []
        if i % 2 == 0:
            if num1 == len(splita):
                break
            temp.append(splita[num1])
            num1 += 1
        else:
            if num2 == len(splitb):
                break
            temp.append(splitb[num2])
            num2 += 1

        if i == len(splitc):
            break
        temp.append(splitc[i + 1])
        if i == len(splitd):
            break
        temp.append(splitd[i + 1])
        temp.append(splite[i + 1])
        keyvalpair.append(temp)

    return keyvalpair


# number is the keyvalue pair row index, 0 = names of variables,1=s,2=r,3=1
def splitData(matrixnumber, number, data):
    keyValue = splitKeyValue(data)
    onegivenSeconds[matrixnumber] = keyValue[0][number]
    onegivenLevel[matrixnumber] = keyValue[1][number]
    onegivenVariables[matrixnumber] = keyValue[2][number]
    onegivenUsed[matrixnumber] = keyValue[3][number]
    onegivenOriginal[matrixnumber] = keyValue[4][number]
    onegivenConflicts[matrixnumber] = keyValue[5][number]
    onegivenLeanrned[matrixnumber] = keyValue[6][number]
    onegivenLimit[matrixnumber] = keyValue[7][number]
    onegivenAgility[matrixnumber] = keyValue[8][number]
    onegivenMB[matrixnumber] = keyValue[9][number]


def splitsysData(j, given, number, data):
    keyValue = splitKeyValue(data)
    symgivenSeconds[j, given] = keyValue[0][number]
    symgivenLevel[j, given] = keyValue[1][number]
    symgivenVariables[j, given] = keyValue[2][number]
    symgivenUsed[j, given] = keyValue[3][number]
    symgivenOriginal[j, given] = keyValue[4][number]
    symgivenConflicts[j, given] = keyValue[5][number]
    symgivenLeanrned[j, given] = keyValue[6][number]
    symgivenLimit[j, given] = keyValue[7][number]
    symgivenAgility[j, given] = keyValue[8][number]
    symgivenMB[j, given] = keyValue[9][number]


# out is the verbose split data 0 = names of variables,1=s,2=r,3=1
def solve(given, outtype, filenumber, symmetry, type2):
    j = 0
    while j < 100:  # change here for 1000
        f = open("output_Data" + str(filenumber) + ".csv", 'a')
        if symmetry == 0 and type2 == 0:
            probnumber = (given * 100) + j  # change here for 1000
        else:
            probnumber = (j * 100) + given
        prob, text = getProblem(probnumber, symmetry,type2)
        f.write("problem:, " + str(prob) + "\n")
        f.write("Symmetry applied:, " + text + "\n")
        cnf = get_rules(9) + prob
        with open('data.txt', 'w') as outfile:
            json.dump(cnf, outfile)
        p = subprocess.Popen(["python", "Pycosat_runner.py"], shell=True, stdout=subprocess.PIPE)
        pycoutput = p.communicate()[0]
        outverbose = find_between(str(pycoutput), first, last)
        outversplit = outverbose.split('\\nc')
        f.write(str(given) + "\n")
        keyval = splitKeyValue(outversplit)
        for col in range(0, 4):
            for row in range(len(keyval)):
                f.write(keyval[row][col] + ",")
            f.write('\n')
        f.write("restarts: " + str(str(pycoutput).count('C R')))
        f.write('\n')
        f.close()
        if symmetry == 0 and type2 == 0:
            splitData(j, outtype, outversplit)
        else:
            splitsysData(j, given, outtype, outversplit)
        j += 1
    if symmetry == 0 and type2 == 0:
        plotonegiven(j, given)
    return


def plotgraph1(x, y, filename):
    plt.plot(np.linspace(1, x, x, endpoint=True), y, color='red')
    plt.xticks(np.linspace(1, x, 10, endpoint=True))
    plt.savefig(filename)
    plt.close()


def plotgraph2(x, a, b, c, filename):
    plt.plot(np.linspace(1, x, x, endpoint=True), a, color='red')
    plt.plot(np.linspace(1, x, x, endpoint=True), b, color='green')
    plt.plot(np.linspace(1, x, x, endpoint=True), c, color='blue')
    plt.xticks(np.linspace(1, x, 10, endpoint=True))
    plt.savefig(filename)
    plt.close()


def plotonegiven(x, filenumber):
    fnum = str(filenumber)
    plotgraph1(x, onegivenLevel, "onegivenLevel" + fnum)
    plotgraph1(x, onegivenVariables, "onegivenVariables" + fnum)
    plotgraph1(x, onegivenAgility, "onegivenAgility" + fnum)
    plotgraph1(x, onegivenConflicts, "onegivenConflicts" + fnum)
    plotgraph1(x, onegivenLeanrned, "onegivenLeanrned" + fnum)
    plotgraph1(x, onegivenLimit, "onegivenLimit" + fnum)
    plotgraph1(x, onegivenMB, "onegivenMB" + fnum)
    plotgraph1(x, onegivenOriginal, "onegivenOriginal" + fnum)
    plotgraph1(x, onegivenSeconds, "onegivenSeconds" + fnum)
    plotgraph1(x, onegivenUsed, "onegivenUsed" + fnum)


def plotsymgiven(x, y, filenumber):
    fnum = str(filenumber)
    plotgraph1(x, symgivenLevel[y], "symgivenLevel" + fnum)
    plotgraph1(x, symgivenVariables[y], "symgivenVariables" + fnum)
    plotgraph1(x, symgivenAgility[y], "symgivenAgility" + fnum)
    plotgraph1(x, symgivenConflicts[y], "symgivenConflicts" + fnum)
    plotgraph1(x, symgivenLeanrned[y], "symgivenLeanrned" + fnum)
    plotgraph1(x, symgivenLimit[y], "symgivenLimit" + fnum)
    plotgraph1(x, symgivenMB[y], "symgivenMB" + fnum)
    plotgraph1(x, symgivenOriginal[y], "symgivenOriginal" + fnum)
    plotgraph1(x, symgivenSeconds[y], "symgivenSeconds" + fnum)
    plotgraph1(x, symgivenUsed[y], "symgivenUsed" + fnum)


def plotsymaverage(x, filenumber):
    fnum = str(filenumber)
    plotgraph2(x, avggivenLevel, maxgivenLevel, mingivenLevel, "allgivenLevel" + fnum)
    plotgraph2(x, avggivenVariables, maxgivenVariables, mingivenVariables, "allgivenVariables" + fnum)
    plotgraph2(x, avggivenAgility, maxgivenAgility, mingivenAgility, "allgivenAgility" + fnum)
    plotgraph2(x, avggivenConflicts, maxgivenConflicts, mingivenConflicts, "allgivenConflicts" + fnum)
    plotgraph2(x, avggivenLeanrned, maxgivenLeanrned, mingivenLeanrned, "allgivenLeanrned" + fnum)
    plotgraph2(x, avggivenLimit, maxgivenLimit, mingivenLimit, "allgivenLimit" + fnum)
    plotgraph2(x, avggivenMB, maxgivenMB, mingivenMB, "allgivenMB" + fnum)
    plotgraph2(x, avggivenOriginal, maxgivenOriginal, mingivenOriginal, "allgivenOriginal" + fnum)
    plotgraph2(x, avggivenSeconds, maxgivenSeconds, mingivenSeconds, "allgivenSeconds" + fnum)
    plotgraph2(x, avggivenUsed, maxgivenUsed, mingivenUsed, "allgivenUsed" + fnum)


def plotsymaverageonly(x, filenumber):
    fnum = str(filenumber)
    plotgraph1(x, avggivenLevel, "avggivenLevel" + fnum)
    plotgraph1(x, avggivenVariables, "avggivenVariables" + fnum)
    plotgraph1(x, avggivenAgility, "avggivenAgility" + fnum)
    plotgraph1(x, avggivenConflicts, "avggivenConflicts" + fnum)
    plotgraph1(x, avggivenLeanrned, "avggivenLeanrned" + fnum)
    plotgraph1(x, avggivenLimit, "avggivenLimit" + fnum)
    plotgraph1(x, avggivenMB, "avggivenMB" + fnum)
    plotgraph1(x, avggivenOriginal, "avggivenOriginal" + fnum)
    plotgraph1(x, avggivenSeconds, "avggivenSeconds" + fnum)
    plotgraph1(x, avggivenUsed, "avggivenUsed" + fnum)


def populatesymavg(given):
    avggivenVariables[given] = np.average(symgivenVariables[:,given])
    avggivenOriginal[given] = np.average(symgivenOriginal[:,given])
    avggivenLeanrned[given] = np.average(symgivenLeanrned[:,given])
    avggivenAgility[given] = np.average(symgivenAgility[:,given])
    avggivenLevel[given] = np.average(symgivenLevel[:,given])
    avggivenUsed[given] = np.average(symgivenUsed[:,given])
    avggivenSeconds[given] = np.average(symgivenSeconds[:,given])
    avggivenConflicts[given] = np.average(symgivenConflicts[:,given])
    avggivenLimit[given] = np.average(symgivenLimit[:,given])
    avggivenMB[given] = np.average(symgivenMB[:,given])


def populatesymmax(given):
    maxgivenVariables[given] = np.max(symgivenVariables[:,given])
    maxgivenOriginal[given] = np.max(symgivenOriginal[:,given])
    maxgivenLeanrned[given] = np.max(symgivenLeanrned[:,given])
    maxgivenAgility[given] = np.max(symgivenAgility[:,given])
    maxgivenLevel[given] = np.max(symgivenLevel[:,given])
    maxgivenUsed[given] = np.max(symgivenUsed[:,given])
    maxgivenSeconds[given] = np.max(symgivenSeconds[:,given])
    maxgivenConflicts[given] = np.max(symgivenConflicts[:,given])
    maxgivenLimit[given] = np.max(symgivenLimit[:,given])
    maxgivenMB[given] = np.max(symgivenMB[:,given])


def populatesymmin(given):
    mingivenVariables[given] = np.min(symgivenVariables[:,given])
    mingivenOriginal[given] = np.min(symgivenOriginal[:,given])
    mingivenLeanrned[given] = np.min(symgivenLeanrned[:,given])
    mingivenAgility[given] = np.min(symgivenAgility[:,given])
    mingivenLevel[given] = np.min(symgivenLevel[:,given])
    mingivenUsed[given] = np.min(symgivenUsed[:,given])
    mingivenSeconds[given] = np.min(symgivenSeconds[:,given])
    mingivenConflicts[given] = np.min(symgivenConflicts[:,given])
    mingivenLimit[given] = np.min(symgivenLimit[:,given])
    mingivenMB[given] = np.min(symgivenMB[:,given])


def plotaverage(x, filenumber):
    fnum = str(filenumber)
    plotgraph2(x, avggivenLevel, maxgivenLevel, mingivenLevel, "allgivenLevel" + fnum)
    plotgraph2(x, avggivenVariables, maxgivenVariables, mingivenVariables, "allgivenVariables" + fnum)
    plotgraph2(x, avggivenAgility, maxgivenAgility, mingivenAgility, "allgivenAgility" + fnum)
    plotgraph2(x, avggivenConflicts, maxgivenConflicts, mingivenConflicts, "allgivenConflicts" + fnum)
    plotgraph2(x, avggivenLeanrned, maxgivenLeanrned, mingivenLeanrned, "allgivenLeanrned" + fnum)
    plotgraph2(x, avggivenLimit, maxgivenLimit, mingivenLimit, "allgivenLimit" + fnum)
    plotgraph2(x, avggivenMB, maxgivenMB, mingivenMB, "allgivenMB" + fnum)
    plotgraph2(x, avggivenOriginal, maxgivenOriginal, mingivenOriginal, "allgivenOriginal" + fnum)
    plotgraph2(x, avggivenSeconds, maxgivenSeconds, mingivenSeconds, "allgivenSeconds" + fnum)
    plotgraph2(x, avggivenUsed, maxgivenUsed, mingivenUsed, "allgivenUsed" + fnum)


def plotaverageonly(x, filenumber):
    fnum = str(filenumber)
    plotgraph1(x, avggivenLevel, "avggivenLevel" + fnum)
    plotgraph1(x, avggivenVariables, "avggivenVariables" + fnum)
    plotgraph1(x, avggivenAgility, "avggivenAgility" + fnum)
    plotgraph1(x, avggivenConflicts, "avggivenConflicts" + fnum)
    plotgraph1(x, avggivenLeanrned, "avggivenLeanrned" + fnum)
    plotgraph1(x, avggivenLimit, "avggivenLimit" + fnum)
    plotgraph1(x, avggivenMB, "avggivenMB" + fnum)
    plotgraph1(x, avggivenOriginal, "avggivenOriginal" + fnum)
    plotgraph1(x, avggivenSeconds, "avggivenSeconds" + fnum)
    plotgraph1(x, avggivenUsed, "avggivenUsed" + fnum)


def populateavg(given):
    avggivenVariables[given] = np.average(onegivenVariables)
    avggivenOriginal[given] = np.average(onegivenOriginal)
    avggivenLeanrned[given] = np.average(onegivenLeanrned)
    avggivenAgility[given] = np.average(onegivenAgility)
    avggivenLevel[given] = np.average(onegivenLevel)
    avggivenUsed[given] = np.average(onegivenUsed)
    avggivenSeconds[given] = np.average(onegivenSeconds)
    avggivenConflicts[given] = np.average(onegivenConflicts)
    avggivenLimit[given] = np.average(onegivenLimit)
    avggivenMB[given] = np.average(onegivenMB)


def populatemax(given):
    maxgivenVariables[given] = np.max(onegivenVariables)
    maxgivenOriginal[given] = np.max(onegivenOriginal)
    maxgivenLeanrned[given] = np.max(onegivenLeanrned)
    maxgivenAgility[given] = np.max(onegivenAgility)
    maxgivenLevel[given] = np.max(onegivenLevel)
    maxgivenUsed[given] = np.max(onegivenUsed)
    maxgivenSeconds[given] = np.max(onegivenSeconds)
    maxgivenConflicts[given] = np.max(onegivenConflicts)
    maxgivenLimit[given] = np.max(onegivenLimit)
    maxgivenMB[given] = np.max(onegivenMB)


def populatemin(given):
    mingivenVariables[given] = np.min(onegivenVariables)
    mingivenOriginal[given] = np.min(onegivenOriginal)
    mingivenLeanrned[given] = np.min(onegivenLeanrned)
    mingivenAgility[given] = np.min(onegivenAgility)
    mingivenLevel[given] = np.min(onegivenLevel)
    mingivenUsed[given] = np.min(onegivenUsed)
    mingivenSeconds[given] = np.min(onegivenSeconds)
    mingivenConflicts[given] = np.min(onegivenConflicts)
    mingivenLimit[given] = np.min(onegivenLimit)
    mingivenMB[given] = np.min(onegivenMB)


def switchrowstrips(npmatrix):
    b = np.copy(npmatrix[0, :])
    c = np.copy(npmatrix[6, :])
    npmatrix[0, :] = c
    npmatrix[6, :] = b
    b = np.copy(npmatrix[1, :])
    c = np.copy(npmatrix[7, :])
    npmatrix[1, :] = c
    npmatrix[7, :] = b
    b = np.copy(npmatrix[2, :])
    c = np.copy(npmatrix[8, :])
    npmatrix[2, :] = c
    npmatrix[8, :] = b
    return npmatrix


def switchcolumnstrips(npmatrix):
    b = np.copy(npmatrix[:, 0])
    c = np.copy(npmatrix[:, 6])
    npmatrix[:, 0] = c
    npmatrix[:, 6] = b
    b = np.copy(npmatrix[:, 1])
    c = np.copy(npmatrix[:, 7])
    npmatrix[:, 1] = c
    npmatrix[:, 7] = b
    b = np.copy(npmatrix[:, 2])
    c = np.copy(npmatrix[:, 8])
    npmatrix[:, 2] = c
    npmatrix[:, 8] = b
    return npmatrix


def switchdiagonally(npmatrix):
    return npmatrix.transpose()


def checkevenandequal(npmatrix):
    a, b = np.unique(npmatrix, return_counts=True)
    c = []
    for i in range(1, len(b) + 1):
        for j in range(i + 1, len(b) + 1):
            if j > len(b) - 1:
                continue
            if b[i] == b[j]:
                if a[i] in [y for x in c for y in x]:
                    continue
                temp = [a[i], a[j]]
                c.append(temp)
    return c


def switchelement(elementlist, npmatrix):
    for i in range(0, npmatrix.shape[0]):
        for j in range(0, npmatrix.shape[1]):
            for k in range(len(elementlist)):
                if npmatrix[i][j] == elementlist[k][0]:
                    npmatrix[i][j] = elementlist[k][1]
                elif npmatrix[i][j] == elementlist[k][1]:
                    npmatrix[i][j] = elementlist[k][0]
    return npmatrix


'''symmetry encoding:
0 or empty  = randomize
1           = row strip switch
2           = column strip switch
3           = row and column switch
4           = diagonal switch
5           = row and diagonal
6           = column and diagonal
7           = row, column and diagonal
8           = element switch
9           = element and row switch
10          = element and column switch
11          = element , row and column switch
12          = element and diagonal switch
13          = element , diagonal and row switch
14          = element , diagonal and column switch
15          = element , diagonal , row and column switch  
'''
'''set type to non zero if you want to analyze the symmetric set without any tranformation'''


def solveall(symmetry=0, type2=0):
    i = 0
    for i in range(0, 81):
        solve(i, 3, i, symmetry, type2)
        if symmetry == 0 & type2 == 0:
            populateavg(i)
            populatemax(i)
            populatemin(i)
        else:
            populatesymavg(i)
            populatesymmax(i)
            populatesymmin(i)
    if symmetry != 0 or type2 !=0:
        for i in range(100):
            plotsymgiven(81, i, i)

    plotaverageonly(81, i)
    plotaverage(81, i)


'''def solve(given,outtype):
    j = 0
    probnumber = (given*100)+j
    prob = getProblem(probnumber)
    cnf = get_rules(9) + prob
    with open('data.txt', 'w') as outfile:
        json.dump(cnf, outfile)
    p = subprocess.Popen(["python", "Pycosat_runner.py"], shell=True, stdout=subprocess.PIPE)
    pycoutput = p.communicate()[0]
    outverbose = find_between(str(pycoutput), first, last)
    outversplit = outverbose.split('\\nc')
    f = open("output_Data.csv", 'a')
    f.write(str(given) + "\n")
    keyval = splitKeyValue(outversplit)
    for col in range(0,4):
        for row in range(len(keyval)):
            f.write(keyval[row][col] + ",")
        f.write('\n')
    f.close()
    splitData(j,outtype,outversplit)
    j+=1'''

'''p = subprocess.Popen(["python", "Pycosat_runner.py"], shell=True, stdout=subprocess.PIPE)
pycoutput = p.communicate()[0]
outverbose = find_between(str(pycoutput), first, last)
outversplit = outverbose.split('\\nc')


furtherSplit =  str(outversplit[0]+outversplit[1])+"\n"+str(outversplit[3])+"\n"+str(outversplit[4]) +"\n"+str(outversplit[8]) +"\n" + str(outversplit[6])'''

'''import json
first2= '-2, -3'
last2= '\\r'
probnumber = 4567
a = np.copy(newMatrix[probnumber])
prob,text = getProblem(probnumber)
f = open("output.txt",'w')
f.write("problem:, " + str(prob) + "\n")
cnf = get_rules(9) + prob
with open('data.txt', 'w') as outfile:
    json.dump(cnf, outfile)
p = subprocess.Popen(["python", "Pycosat_runner.py"], shell=True, stdout=subprocess.PIPE)
pycoutput = p.communicate()[0]
outverbose = find_between(str(pycoutput), first, last)
outversplit = outverbose.split('\\nc')
f.write(str(probnumber) + "\n")
keyval = splitKeyValue(outversplit)
for col in range(0, 4):
        for row in range(len(keyval)):
            f.write(keyval[row][col] + ",")
        f.write('\n')
outverbose2 = find_between(str(pycoutput), first2, last2)
c = outverbose2.split(',')
c.remove('')
if ' -999]' in c:
    c.remove(' -999]')
    c.append(' -999')
elif  ' 999]' in c:
    c.remove(' 999]')
    c.append(' 999')
b = [int(x) for x in c if int(x) > 110]
f.write(str(b))
f.close()
displayMatrix(b)'''
