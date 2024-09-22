import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

from bg_taskabc import MSE, R2, Design_Matrix_2D, regression



def plot_surface():
    surf = imread('Lausanne.tif')   

    # Show the terrain
    plt.figure()
    plt.title('Terrain over Lausanne')
    plt.imshow(surf, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()  

    m = len(surf)

    x = np.linspace(0,1,m)
    y = np.linspace(0,1,m)
    x, y = np.meshgrid(x,y)

    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(1,1,1, projection='3d')

    surf = ax.plot_surface(x, y, surf, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha = 0.7)

    fig.suptitle('Lausanne DTM', fontsize=16)
    ax.set_zlim(-0.10, 10000)

    # Add a color bar which maps values to colors.
    colorbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()






# ------------------------------...................- Make data ----------------------------------------------------

# Other terrain example
# surf = imread('Sao_Tome.tif')

surf = imread('Lausanne.tif')    

# ---------- Slice the terrain model ---------------
row_start = 100 # Start of rows 
row_end = 100 + 30     # End of rows 

col_start = row_start   # Start of rows 
col_end = row_end       # End of rows 

# Extract the center half of the matrix
surf = surf[row_start:row_end, col_start:col_end]
# --------------------------------------------------

z = surf

m = len(surf)
x = np.linspace(0,1,m)
y = np.linspace(0,1,m)
x, y = np.meshgrid(x,y)

deg_max = 10

print(x.shape, y.shape, z.shape)
print(z)
regression(x,y,z,deg_max,bool_info=True)

