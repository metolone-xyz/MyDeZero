import numpy as np
import dezero.functions as F
import dezero.functions_conv as FC

x1 = np.random.rand(1, 3, 7, 7)
col1 = FC.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)