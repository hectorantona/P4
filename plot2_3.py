import numpy as np
import glob
import matplotlib.pyplot as mpl

def get_d(path):
    all_files = glob.glob(path)
    c = []
    for filename in all_files:
        str = open(filename, 'r').read()
        str = str.split('\n')
        str = str[1:-3]
        for i in range(0,len(str)):
            aux = str[i].split('\t')
            str[i] = float(aux[1])
        c.extend(str)
    return c

c2_lp = get_d('./Datos/lp/c2/*.txt')
c3_lp = get_d('./Datos/lp/c3/*.txt')

c2_lpcc = get_d('./Datos/lpcc/c2/*.txt')
c3_lpcc = get_d('./Datos/lpcc/c3/*.txt')

c2_mfcc = get_d('./Datos/mfcc/c2/*.txt')
c3_mfcc = get_d('./Datos/mfcc/c3/*.txt')

'''
mpl.title("LP")
mpl.scatter(c2_lp, c3_lp, s=0.42)
mpl.xlabel('C2')
mpl.ylabel('C3')
mpl.show()

mpl.title("LPCC")
mpl.scatter(c2_lpcc, c3_lpcc, s=0.42)
mpl.xlabel('C2')
mpl.ylabel('C3')
mpl.show()

mpl.title("MFCC")
mpl.scatter(c2_mfcc, c3_mfcc, s=0.42)
mpl.xlabel('C2')
mpl.ylabel('C3')
mpl.show()
'''

x = np.corrcoef(c2_mfcc, c3_mfcc)
print(x)



