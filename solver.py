import itertools
import sys
import math
import pycosat
import numpy as np


class solver(object):
    def __init__(self):
        self.cnf = []

    def get_rowrules(self, M):
        """
        get_rowrules returns the rules that make sure no duplicates can
        be in one row.
        """
        f = open("result.txt", 'a')
        rangem = np.arange(1,M+1,1)
        rowrule = []
        x = ""
        y = ""
        for row in rangem:
            for val in rangem:
                temp1 = []
                y = "\n"
                for i in np.arange(1, M + 1, 1):
                    temp1.append((row * 100) + ( i* 10) + val, )
                    y = str((row * 100) + ( i* 10) + val)
                if len(temp1) != 0:
                    rowrule.append(temp1)
                for col in rangem:
                    for j in np.arange(col, M + 1, 1):
                        temp = []
                        if col == j:
                            continue
                        temp.append(-((row * 100) + (col * 10) + (val)))
                        temp.append(-((row * 100) + (j * 10) + (val)))
                        x += "-" + str((row * 100) + (col * 10) + (val)) + " \t : " + "-" + str((row * 100) + (j * 10) + (val)) + "\n"
                        rowrule.append(temp)
        f.write(y)
        f.write(x)
        f.close()
        return rowrule

    def get_columnrules(self, M):
        """
        get_columnrules returns the rules that make sure no duplicated can
        be in a column.
        """
        f = open("result.txt", 'a')
        x = ""
        y = ""
        rangem = np.arange(1, M + 1, 1)
        colrule = []
        for col in rangem:
            for val in rangem:
                temp1 = []
                y = "\n"
                for i in np.arange(1, M + 1, 1):
                    temp1.append((i * 100) + (col * 10) + val, )
                    y = str((i * 100) + (col * 10) + val) + "\t"
                if len(temp1) != 0:
                    colrule.append(temp1)
                for row in rangem:
                    for j in np.arange(row, M + 1, 1):
                        temp = []
                        if row == j:
                            continue
                        temp.append(-((row * 100) + (col * 10) + (val)))
                        temp.append(-((j * 100) + (col * 10) + (val)))
                        colrule.append(temp)
                        x += "-" + str((row * 100) + (col * 10) + (val)) + " \t : " + "-" + str((j * 100) + (col * 10) + (val)) + "\n"

        f.write(y)
        f.write(x)
        f.close()
        return colrule

    def get_boxrules(self, M):
        """
        get_boxrules returns the rules that make sure there can be no
        duplicates in one box.
        """
        N=M
        sq = 3
        a= int(N/sq)
        boxrules = []
        rangeM= np.arange(1, M + 1)
        f = open("result.txt", 'w')
        x = ""
        y = ""
        num1 = 1
        for s in range(1, a+1): #1 to 3
            for t in range(1, a+1): #1 to 3
                for r in range((s * (a))-2, ((s + 1) * (a))-2): #1-4:4-7:7-10
                    for c in range((t * (a))-2, ((t + 1) * (a))-2):
                        temp2 = []
                        if num1 == 10:
                            num1 = 1
                        for r2 in range((s * (a))-2, ((s + 1) * (a))-2):
                            for c2 in range((t * (a))-2, ((t + 1) * (a))-2):
                                # comment this out to see if your python doesn't suck.
                                # it doesn't error iff it sucks
                                f = open("result.txt", 'a')
                                for i in range(1, N+1):
                                    temp =[]
                                    if r != r2 and c != c2:
                                        temp.append(-((r * 100) + (c * 10) + (i)))
                                        temp.append(-((r2 * 100) + (c2 * 10) + (i)))
                                    if len(temp) == 0:
                                        continue
                                    x += "-" + str((r * 100) + (c * 10) + (i)) + " \t : " + "-" + str((r2 * 100) + (c2 * 10) + (i)) + "\n"

                                    boxrules.append(temp)

                                num = ((r2 * 100) + (c2 * 10) + (num1))

                                if num in temp2:
                                    continue
                                temp2.append(num)
                                y += str(num) + "\t"
                        y += "\n"
                        boxrules.append(temp2)
                        temp2 = []
                        num1 +=1
        f.write(y)
        f.write(x)
        f.close()
        return boxrules

    def at_least_one_rules( M):
        """
        at_least_one_rule returns the rules that make sure there has
        to be at least one number in a square.
        """
        f = open("result.txt", 'a')
        y = ""
        rangem = range(1, M + 1, 1)
        atleastrule = []
        for row in rangem:
            for col in rangem:
                temp = []
                for val in rangem:
                    temp.append((row * 100) + (col * 10) + val, )
                    y = str((row * 100) + (col * 10) + val) + "\t"
                atleastrule.append(temp)
        f.write(y)
        f.close()
        return atleastrule

    def get_rules(self, M):
        """
        get_rules returns all the suduko rules of a sudo of size M.
        """
        self.cnf.append(get_rowrules(9))
        self.cnf.append(get_columnrules(9))
        self.cnf.append(get_boxrules(M))
        return self.cnf

    def solve(self, sudoku, verbose):
        return [x for x in pycosat.solve(self.get_rules(9) + sudoku,verbose) if x > 0]

    '''def solve2(self, sudoku):
        return [x for x in pycosat.itersolve(self.get_rules(9) + sudoku) if x > 0]'''