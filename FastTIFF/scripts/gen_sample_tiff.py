import numpy as np
import tifffile


# matrix = np.random.rand(4, 512, 512).astype(np.float64)
matrix = np.random.randint(0, 100, (3, 512, 512), dtype=np.int8)


tifffile.imwrite('example.tif', matrix)
