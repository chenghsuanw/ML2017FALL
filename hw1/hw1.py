import pandas as pd
import numpy as np
import sys

test_csv = pd.read_csv(sys.argv[1],encoding='big5',header=None)
test_csv.replace("NR", 0, inplace=True)
test_raw = pd.DataFrame.as_matrix(test_csv)
test_raw = np.delete(test_raw,range(2),1)
test_raw = test_raw.astype(float)

weight = np.loadtxt("weight.txt")
bias = np.loadtxt("bias.txt")

with open(sys.argv[2], 'w') as f:
	f.write("id,value\n")
	for i in range(240):
		test_data = np.array(test_raw[i*18:(i+1)*18]).T
		for j in range(18):
			test_data = np.column_stack((test_data, test_data[:,j]*test_data[:,9]))
		y = np.sum(weight * test_data) + bias
		f.write("id_"+str(i)+","+str(y)+"\n")