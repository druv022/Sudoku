import pycosat
import json

verbose = 5

json_data=open("data.txt").read()
cnf = json.loads(json_data)

a = pycosat.solve(cnf, verbose = 5)

print(a)
