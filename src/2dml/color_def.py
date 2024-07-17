# colormap definitions

#UPODATED

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def makecolorBR3():
    """ define colormap """

    cdict3 = {'red':  ((0.0, 0.05, 0.05),
                       (0.10, 0.05, 0.1),
                       (0.15, 0.1, 0.05),
                       (0.20, 0.2, 0.2),
                       (0.30, 0.1, 0.1),
                       (0.45, 0.0, 0.0),
                       (0.49, 0.0, 0.0),
                       (0.65, 0.0, 0.0),
                       (0.68, 0.0, 0.0), # setting this to zero removes pink area
                       (0.70, 1.0, 1.00),
                       (0.74, 1.0, 1.0),
                       (0.80, 0.9, 0.9),
                       (0.90, 0.7, 0.7),
                       (1.0, 0.5, 0.5)),

             'green': ((0.0, 0.05, 0.05),
                       (0.10, 0.05, 0.05),
                       (0.15, 0.1, 0.1),
                       (0.20, 0.2, 0.2),
                       (0.30, 0.2, 0.2),
                       (0.45, 0.4, 0.4),
                       (0.49, 0.4, 0.4),
                       (0.65,  0.6, 0.6),
                       (0.68,  0.8, 0.8),
                       (0.70, 1.0, 1.0),
                       (0.74, 1.0, 1.0),
                       (0.80, 0.3, 0.3),
                       (0.90, 0.2, 0.2),
                       (1.0, 0.1, 0.1)),

             'blue':  ((0.0, 0.1, 0.1),
                       (0.10, 0.1, 0.1),
                       (0.15, 0.2, 0.2),
                       (0.20, 0.4, 0.4),
                       (0.30, 0.5, 0.5),
                       (0.45, 0.6, 0.6),
                       (0.49, 0.7, 0.7),
                       (0.65,  0.7, 0.7),
                       (0.68,  0.8, 0.7),
                       (0.70, 1.0, 1.0),
                       (0.74, 1.0, 1.0),
                       (0.80, 0.3, 0.3),
                       (0.90, 0.2, 0.2),
                       (1.0, 0.1, 0.1))
            }

    plt.register_cmap(name='BlueRed3', data=cdict3, lut=128)  # optional lut kwarg

    my_cmap = plt.get_cmap('BlueRed3')

    plt.rcParams['image.cmap'] = 'BlueRed3'

    return my_cmap
