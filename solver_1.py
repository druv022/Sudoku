import itertools
import sys
import math
import pycosat


class solver1(object):
    def __init__(self):
        self.matrix = []


    def strip_else(self, x, y, number):
        """
        strip returns the negative value of x, y and number appended.
        """
        return int('-' + str(int(x)) + str(int(y)) + str(number))

    def get_unaryrules_else(self, index, M):
        """
        get_unaryrules returns the rules that make sure in one square
        there can be only one number for index.
        """
        return list(
            itertools.chain.from_iterable(
                [[[y, self.strip(index[0], index[1], x)] for x in range(1, M+1)
                  if y != self.strip(index[0], index[1], x)] for y in
                 [self.strip(index[0], index[1], i) for i in range(1, M+1)]]
            ))

    def at_least_one_rules_else(self, index, M):
        """
        at_least_one_rule returns the rules that make sure there has
        to be at least one number in a square.
        """
        return [[int(str(index[0]) + str(index[1]) + str(x))
                for x in range(1, M+1)]]

    def get_rowrules_else(self, index, M):
        """
        get_rowrules returns the rules that make sure no duplicates can
        be in one row.
        """
        return list(itertools.chain.from_iterable(
            [[[self.strip(index[0], index[1], i), self.strip(x, index[1], i)]
              for x in range(1, M+1) if x != index[0]]
                for i in range(1, M+1)]
        ))

    def get_columnrules_else(self, index, M):
        """
        get_columnrules returns the rules that make sure no duplicated can
        be in a column.
        """
        return list(itertools.chain.from_iterable(
            [[[self.strip(index[0], index[1], i), self.strip(index[0], x, i)]
              for x in range(1, M+1) if x != index[1]]
                for i in range(1, M+1)]
        ))

    def lower_index_else(self, x, index, M):
        """
        lower_index returns the index warped down to the bottom left square of
        the square index is in.
        """
        return [x[0]+index[0]-((index[0]-1) % math.sqrt(M)),
                x[1]+index[1]-((index[1]-1) % math.sqrt(M))]

    def get_boxrules_else(self, index, M):
        """
        get_boxrules returns the rules that make sure there can be no
        duplicates in one box.
        """
        return list(itertools.chain.from_iterable(
            [[[self.strip(index[0], index[1], z), self.strip(l[0], l[1], z)]
              for z in range(1, M+1)] for l in
             [self.lower_index(x, index, M) for x in
              itertools.product(range(0, int(math.sqrt(M))),
                                range(0, int(math.sqrt(M))))
              if self.lower_index(x, index, M) != [index[0], index[1]]]]
        ))

    def get_rules_else(self, M):
        """
        get_rules returns all the suduko rules of a sudo of size M.
        """
        return list(itertools.chain.from_iterable(
            list(itertools.chain.from_iterable(
                [[self.get_boxrules(x, M), self.get_unaryrules(x, M),
                  self.at_least_one_rules(x, M), self.get_columnrules(x, M),
                  self.get_rowrules(x, M)] for x in itertools.product(range(1, M+1),
                                                                      range(1, M+1))]
            ))
        ))

    def solve_else(self, sudoku, verbose):
        return [x for x in pycosat.solve(self.get_rules(9) + sudoku,verbose) if x > 0]

    '''def solve2(self, sudoku):
        return [x for x in pycosat.itersolve(self.get_rules(9) + sudoku) if x > 0]'''