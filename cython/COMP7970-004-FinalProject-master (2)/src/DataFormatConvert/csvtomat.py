import scipy.io as sio
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", help="filename")
parser.add_argument("-o", help="output")
args = parser.parse_args()
fileName = args.f
outputFile = args.o
if fileName is None:
    raise ValueError('invalid input file')
if outputFile is None:
    outputFile = 'output.mat'

data = pd.read_csv(fileName, header =None)
data_size = len(data.get_values())
sparse_row = np.zeros(data_size * 2,dtype=int)
sparse_column = np.zeros(data_size * 2,dtype=int)
sparse_data = np.ones(data_size * 2)
count = 0
for node_1,node_2 in data.get_values():
    # make nodes starts from 0
    new_node_1 = node_1 - 1
    new_node_2 = node_2 - 1
    sparse_row[count] = new_node_1
    sparse_column[count] = new_node_2
    count += 1
    sparse_row[count] = new_node_2
    sparse_column[count] = new_node_1
    count += 1
sio.savemat(outputFile,{'network':csr_matrix((sparse_data, (sparse_row, sparse_column)))})
