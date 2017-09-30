import csv
import copy
import numpy as np
import linecache
import subprocess
from random import randint


def file_length(filename):
    with open(filename) as f:
        return sum(1 for line in f)


def rand_stringbase(filename, filelength):
    line = randint(1, filelength)
    return linecache.getline(filename, line)


def det_stringbase(filename, i):
    return linecache.getline(filename, i)


if __name__ == '__main__':
    # pycosat by Tjark Weber
    import webber

    # database for sudokus
    inputfile = "../database/stringbase.txt"
    # output file
    new_csv = "../outputcsv"
    outputfile = open(new_csv, "w")
    writer = csv.writer(outputfile, delimiter=',')

    writer.writerow(
        ['input', 'sat', 'output', 'runtime', 'variables', 'original', 'learned', 'agility', 'level', 'used',
         'conflicts', 'limit', 'mb', 'r_sum'])

    inputfilelength = file_length(inputfile)

    potter = []
    for i in range(1, 1 + inputfilelength):
        inputfilestring = "1" + str(det_stringbase(inputfile, i))
        proc = subprocess.Popen(["python", "-c", "import webber; webber.solve_string(%s)" % (inputfilestring)],
                                stdout=subprocess.PIPE)
        out = proc.communicate()[0]

        output = str(out.upper()).split("\\n")

        r_sum = 0
        restardcounter = 0
        statisticspycosat = []
        for line in output:
            if line[:3] == "C R":
                r_sum += 1
            if restardcounter == 2:
                statisticspycosat.append(str(line))
                break
            if restardcounter == 1:
                statisticspycosat.append(str(line))
                restardcounter = 2
            if line[:3] == "C 1":
                line = " ".join(line[3:].split()).split(" ")[1:]
                line.append(str(r_sum))
                statisticspycosat = line
                restardcounter = 1

        outputfilestring = [inputfilestring[1:][:-2]]

        outputfilestring.append("1")
        outputfilestring.append(statisticspycosat[11])
        outputfilestring.append(statisticspycosat[10])
        for j in range(10):
            outputfilestring.append(statisticspycosat[j])

        print(i)
        writer.writerow(outputfilestring)

