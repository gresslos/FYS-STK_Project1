import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

np.random.seed(1)

def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def FrankeFunction(x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

def Design_Matrix_2D(deg, X):
    # The number of polynomial terms for two variables (x, y) up to degree d is (d+1)(d+2)/2  
    # Minus 1 from dropping intercept-column
    num_terms = int((deg + 1) * (deg + 2) / 2 - 1)

    Phi = np.zeros((X.shape[0], num_terms))
    # PS: not include intercept in design matrix, will scale (centered values)
    col = 0
    for dx in range(1,deg + 1):
        for dy in range(dx + 1):
            # X[:,0] = x-values
            # X[:,1] = y-values
            Phi[:,col] = (X[:,0] ** (dx - dy)) * (X[:,1] ** dy)
            col += 1
    return Phi


def test_1D():
    fig, ax = plt.subplots()

    # Make data.
    x = np.arange(0, 1, 0.05)
    y = 1 # Testing one dimentional
    


    z = FrankeFunction(x, y)

    # Making Design Matrix
    deg = 5 #degree of polynomial

    X = np.zeros((len(x), deg))
    # PS: not include intercept in design matrix, will scale (centered values)
    col = 0
    for dx in range(1,deg + 1):
        X[:,col] = (x ** dx)
        col += 1
        

    # Split data in training and test data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25) 
    # give: test 1/4 of data,  train = 3/4 af data


    # Model Training
    X_train_mean = np.mean(X_train)
    z_train_mean = np.mean(z_train)
    X_train_scaled = X_train - X_train_mean
    z_train_scaled = z_train - z_train_mean
    # Model predictions
    X_test_scaled = X_test - X_train_mean



    # ------------------------- MANUALLY OLS ---------------------------------------------
    # Note: must use PINV, given 1st row = 0, which give det(X) = 0 -> X is Singular Matrix
    # Note: use psudoinverse 
    OLSbeta = np.linalg.pinv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train_scaled
    z_train_OLS = X_train_scaled @ OLSbeta + z_train_mean # adding mean_value

    z_test_OLS = X_test_scaled @ OLSbeta + z_train_mean # adding mean_value


    # Sort the training and test data and corresponding predictions
    sorted_indices = np.argsort(X_train_scaled[:,0].flatten())
    X_train_sort = X_train_scaled[sorted_indices]
    z_train_OLS_sort = z_train_OLS[sorted_indices]
    z_train_sort = z_train_scaled[sorted_indices]

    sorted_indices = np.argsort(X_test_scaled[:,0].flatten())
    X_test_sort = X_test_scaled[sorted_indices]
    z_test_OLS_sort = z_test_OLS[sorted_indices]
    z_test_sort = z_test[sorted_indices]

    # Finding values for plotting (Fist column in Design-Matrix X)
    # Adding mean-value for plotting
    x_train = X_train_sort[:,0] + X_train_mean 
    x_test = X_test_sort[:,0] + X_train_mean


    ax.set_title(f'OLS Regression, Polynomial Degree = {deg}')

    # Plot the surface.
    surf = ax.plot(x, z, label='FrankeFunction') 
    reg = ax.plot(x_train, z_train_OLS_sort, label='OLS Regression')

    ax.legend()
    ax.set_ylim(-0.10, 1.40)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


    # Add a color bar which maps values to colors.
    plt.savefig('FrankeFunction_1D_regression.png')
    plt.show()



    # ----------------------------- MSE ---------------------
    MSETrain_OLS = MSE(z_train_sort + z_train_mean, z_train_OLS_sort)
    MSETest_OLS = MSE(z_test_sort, z_test_OLS_sort)

    R2Train_OLS = R2(z_train_sort + z_train_mean, z_train_OLS_sort)
    R2Test_OLS = R2(z_test_sort, z_test_OLS_sort)

    print(f'Polynomial Degree {deg}')
    print(f"Training MSE for OLS = {MSETrain_OLS:.2e} \nTest MSE OLS = {MSETest_OLS:.2e} \n")
    print(f"Training R2 for OLS = {R2Train_OLS:.2e} \nTest R2 OLS = {R2Test_OLS:.2e}")


def test_2D():

    # Make data.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x,y)

    z = FrankeFunction(x, y)


    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()


    X = np.vstack((x_flat,y_flat)).T
    # X[:,0] = x-vals  X[:,1] = y-vals

    X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.25)

    #----------------------------------------   Scaling -----------------------------
    stdsc = StandardScaler() # For x- and y-vals
    X_train = stdsc.fit_transform(X_train)
    X_test = stdsc.transform(X_test)

    stdsc_z = StandardScaler() # For z-vals
    z_train = stdsc_z.fit_transform(z_train.reshape(-1,1))
    z_test = stdsc_z.transform(z_test.reshape(-1,1))
    #z_train -= z_train_mean


    # Design Matrix Phi
    deg = 5 #degree of polynomial
    Phi_train = Design_Matrix_2D(deg, X_train)
    Phi_test = Design_Matrix_2D(deg, X_test)


    #-----------------------------------------   OLS    -----------------------------
    OLSbeta = np.linalg.inv(Phi_train.T @ Phi_train) @ Phi_train.T @ z_train
    z_pred = Phi_test @ OLSbeta
    z_tilde = Phi_train @ OLSbeta

    #----------------------------------------- Plotting -------------------
    # -----Reshape for plotting
    # -----Reverse Scaling with StandardScaler.inverse_transform()
    X_test = stdsc.inverse_transform(X_test)
    x_test = X_test[:,0].reshape((5,20)) #+ X_train_mean 
    y_test = X_test[:,1].reshape((5,20)) #+ X_train_mean 

    z_pred = stdsc_z.inverse_transform(z_pred)
    z_pred = z_pred.reshape((5,20)) #+ z_train_mean 
    

    X_train = stdsc.inverse_transform(X_train)
    x_train = X_train[:,0].reshape((15,20)) #+ X_train_mean 
    y_train = X_train[:,1].reshape((15,20)) #+ X_train_mean

    z_tilde = stdsc_z.inverse_transform(z_tilde) 
    z_tilde = z_tilde.reshape((15,20)) #+ z_train_mean 


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, alpha = 0.7)

    ax.scatter(x_test, y_test, z_pred, color='r', s=10, label='z_Test (Predicted)')
    ax.scatter(x_train, y_train, z_tilde, color='g', s=10, label='z_Train')

    fig.suptitle('OLS Regression', fontsize=16)
    ax.set_title(f'Polynomial Degree {deg}', fontsize=10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_zlim(-0.10, 1.40)
    ax.legend()

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('test.png')
    plt.show()



    
    # ----------------------------- MSE ---------------------
    # Rescale with StandardScaler
    # Reshape z_train and z_test
    z_test = stdsc_z.inverse_transform(z_test)
    z_test = z_test.reshape((5,20))
    
    z_train = stdsc_z.inverse_transform(z_train)
    z_train = z_train.reshape((15,20)) 
    

    MSETrain_OLS = MSE(z_train, z_tilde)
    MSETest_OLS = MSE(z_test, z_pred)

    R2Train_OLS = R2(z_train, z_tilde)
    R2Test_OLS = R2(z_test, z_pred)

    print(f'      OLS Regression \n   Polynomial Degree {deg}\n')
    print(f"Training MSE for OLS = {MSETrain_OLS:10.2e} \nTest MSE OLS = {MSETest_OLS:18.2e} \n")
    print(f"Training R2 for OLS = {R2Train_OLS:10.2e} \nTest R2 OLS = {R2Test_OLS:18.2e}")
    


test_2D()