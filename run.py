from kmeans import kmeans
from SparseMatrix import SparseMatrix
import argparse
import time
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", help="k_cluster", type = int)
    parser.add_argument("-f", help="filename")
    args = parser.parse_args()
    print args
    fileName = args.f
    k_cluster = args.k
    if fileName is None:
        raise ValueError("main: empty filename")
    elif k_cluster == None or k_cluster <=0:
        raise ValueError("main: invalid number k_cluster")
    else:
        st = time.time()
        s = SparseMatrix(fileName = fileName)
        k = kmeans(k_cluster= 20).fit(s).solve()
        print 'Time used:',str(time.time()-st)
        #print edges.getDataSet()
        #network -> nodes

if __name__ == '__main__':
    main()
