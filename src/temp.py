from Distance import SimMutual
from ReadData import read_from_csv

# read data
data = read_from_csv('data/yang_human_embryo.csv')

X = data.T[:90, 1:]
print(X.shape)

print(SimMutual(X))
