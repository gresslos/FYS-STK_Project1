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

from scipy.ndimage import gaussian_filter

from bg_taskabc import MSE, R2, Design_Matrix_2D, regression
from op_task_f import crossvalidation, crossvalidation_NoLasso


def filter_dtm(data, threshold=0, sigma=1):
    """
    Filters DTM data to remove negative values and apply smoothing.
    
    Parameters:
    - data: numpy array, the DTM data (2D matrix of elevations)
    - threshold: values below this will be set to this threshold
    - sigma: standard deviation for Gaussian smoothing filter (default=1)
    
    Returns:
    - filtered_data: numpy array, the filtered DTM data
    """
    
    # Step 1: Set all values below the threshold (e.g., 0) to the threshold
    data[data < threshold] = threshold
    
    # Step 2: Apply Gaussian smoothing to reduce noise (you can skip this step if unwanted)
    filtered_data = gaussian_filter(data, sigma=sigma)
    
    return filtered_data



def plot_surface(surf):

    # Show the terrain
    plt.figure()
    plt.title('Terrain over Lausanne')
    plt.imshow(surf, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('imshow_real.png')  

    m = len(surf)

    x = np.linspace(0,1,m)
    y = np.linspace(0,1,m)
    x, y = np.meshgrid(x,y)

    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(1,1,1, projection='3d')


    surf = ax.plot_surface(x, y, surf, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha = 0.7)

    fig.suptitle('Lausanne DTM', fontsize=16)
    ax.set_zlim(-0.10, 5000)

    # Add a color bar which maps values to colors.
    colorbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()




# ------------------------------------------------------ Make data ----------------------------------------------------

# Other terrain example
# surf = imread('Sao_Tome.tif')


surf = imread('Surfaces/Lausanne.tif')

# ---------- Slice the terrain model ---------------
row_start = 1000 # Start of rows 
row_end =   1400 # End of rows 

col_start = row_start   # Start of rows 
col_end   = row_end       # End of rows 

# Extract the center half of the matrix
surf = surf[row_start:row_end, col_start:col_end]


#--- Plotting surface --
want_surface = True  # set False not plot surface

if want_surface:
    plot_surface(surf)



z = surf

m = len(surf)
x = np.linspace(0,1,m)
y = np.linspace(0,1,m)
x, y = np.meshgrid(x,y)



# can construct own poly. list 
# PS: list can be of length 1
deg_max = 80
deg_own = [40,45,50,55,60] 
deg_own2 = [55]


print('Regression function is running.')
regression(x,y,z, deg_max, bool_info=True)
#regression(x,y,z,deg_own,bool_info=True)
#regression(x,y,z,deg_own2, bool_info=True)


#can do k=5 since the amount of points is so large
#print('K-fold WITH LASSO is running.')
#crossvalidation(x, y, z, mindeg=deg_own[0], maxdeg=deg_own[-1], interval=5, k=5, a=-8, b=-4)

#can do k=5 since the amount of points is so large
print('K-fold WITHOUT LASSO is running.')
crossvalidation_NoLasso(x, y, z, mindeg=deg_own[0], maxdeg=deg_own[-1], interval=5, k=5, a=-8, b=-4)
