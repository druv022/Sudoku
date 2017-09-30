import numpy as np
from sys import argv
import solver
import load_sudoku


'''def load():
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
    return quizzes, solutions'''

def countGivens(quizzes):
    b= np.array([0])
    i = -1
    while i < 100001:
        i += 1
        if np.count_nonzero(quizzes[i]) in b:
            continue
        b = np.append(b,[np.count_nonzero(quizzes[i])])

    return b


def changeGiven(matrix,givens):

    c,d = np.nonzero(matrix) #c= row, d= column
    e = np.count_nonzero(matrix)
    f = np.shape(c)
    i=0
    while i<(e-givens):
        j = np.random.randint(0,f[0])
        matrix[c[j],d[j]] = 0
        c,d = np.nonzero(matrix)
        f= np.shape(c)
        i+=1
    return matrix

def changeMatrix(matrix1,matrix2):

    i=0
    n = 0
    newQuizMatrix = np.zeros((100000,81),np.int32).reshape((-1,9,9))
    #newQuizMatrix = newQuizMatrix.reshape((-1,9,9))
    while i < 81:
        j = 0
        while j<100:
            a = np.random.randint(0, 100000)
            if (np.count_nonzero(matrix1[a]) < i):
                newQuizMatrix[n] = changeGiven(matrix2[a], i)
            else:
                newQuizMatrix[n] = changeGiven(matrix1[a], i)
            print("Doing i=%s, j=%s" %(i,j))
            j+=1
            n += 1

        i+=1

    return newQuizMatrix





def sudokuProblem(matrix):
    row, col  = np.nonzero(matrix)
    prob = []
    for i in range(row.shape[0]):
        prob.append([100*np.asscalar((row[i]+1))+10*(np.asscalar(col[i]+1))+np.asscalar(matrix[row[i]][col[i]])])
    return prob

def sudokuSolution(list):
    row, col, value = [],[],[]
    for i in list:
        r= (int(i/100))
        c = int((i-r)/10)
        value.append(i-(r+c))
        row.append(r)
        col.append(c)

    return solution


