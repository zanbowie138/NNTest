import idx2numpy
import numpy as np

import matplotlib.pyplot as plt

arr = idx2numpy.convert_from_file('data/t10k-images.idx3-ubyte')
labels = idx2numpy.convert_from_file('data/t10k-labels.idx1-ubyte')

#plt.imshow(arr[4], cmap=plt.cm.binary)
#plt.show()
