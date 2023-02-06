import numpy as np
import math
reallist = np.arange(-math.pow(2,3)+1,math.pow(2,3),2)
imaglist = 1j* (reallist)
cplxlist = reallist + np.transpose(imaglist)
print(cplxlist)
print(np.random.randint(0, 10, [10,1]))