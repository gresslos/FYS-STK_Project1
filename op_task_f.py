import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from random import random, seed
from tqdm import tqdm

#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from bg_taskabc import FrankeFunction, Design_Matrix_2D

#Lasso spits out A LOT of warnings
import warnings
warnings.filterwarnings('ignore')


np.random.seed(1) #set seed for easier troubleshooting



def crossvalidation(x, y, z, mindeg=1, maxdeg=10, interval=1, k=10, a=-6, b=1):
    #Do k-fold cross validation to find the optimal complexity for the model
    #For each complexity, when using Ridge & LASSO regression, the optimal lambda must be found

    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    X = np.vstack((x_flat,y_flat)).T
    # Note: X[:,0] = x_flat  X[:,1] = y_flat
    
    deg = np.arange(mindeg, maxdeg+interval, interval)
    
    #to use GridSearchCV you need a dictionary with parameters
    #appearantly lambda is called alpha in the scikit learn methods
    #according to the documentation the dictionary needs a list and not an array
    alpha_array = np.logspace(a, b, (b-a)+1)
    alpha_list = alpha_array.tolist()
    parameters = {'alpha': alpha_list}
    
    error_test    = np.zeros( (len(deg),3) )
    error_train    = np.zeros( (len(deg),3) )
    
    lmbda_array_Ridge = np.zeros( (len(deg),k) )
    lmbda_array_Lasso = np.zeros( (len(deg),k) )
    
    mse_pred_OLS    = np.zeros(k)
    mse_tilde_OLS   = np.zeros(k)
    mse_pred_Ridge  = np.zeros(k)
    mse_tilde_Ridge = np.zeros(k)
    mse_pred_Lasso  = np.zeros(k)
    mse_tilde_Lasso = np.zeros(k)
    
    
    for i in tqdm( range(len(deg)) ):
        #random_state=1 is for reproducibility, remove the keyword for random 
        #shuffle each time.
        kf = KFold(n_splits=k, shuffle=True)
    
    
        for j, (train, test) in enumerate(kf.split(X)):
            #make the testing and training data from k-fold
            X_train, X_test, z_train, z_test = X[train], X[test], z_flat[train], z_flat[test]
            
            stdsc_z = StandardScaler()
            z_train = stdsc_z.fit_transform( z_train.reshape(-1,1) )
            z_test  = stdsc_z.fit_transform( z_test.reshape(-1,1)  )
            
            #Make the design matrices
            Phi_train = Design_Matrix_2D(deg[i], X_train)
            Phi_test = Design_Matrix_2D(deg[i], X_test)
            
            #scale the design matrices
            stdsc = StandardScaler() # For x- and y-vals
            Phi_train = stdsc.fit_transform(Phi_train)
            Phi_test = stdsc.transform(Phi_test)
            
            
            #calculate the model using the OLS method
            OLSbeta = np.linalg.pinv(Phi_train.T @ Phi_train) @ Phi_train.T @ z_train
            
            #calculate the optimal lambda value using scikit-learn GridSearchCV
            modelRidge = Ridge(fit_intercept=False)
            searchRidge = GridSearchCV(modelRidge, parameters, scoring='neg_mean_squared_error', cv=kf)
            searchRidge.fit(Phi_train, z_train)
            lmbda_array_Ridge[i,j] = searchRidge.best_params_['alpha']
            avg_lmbda_Ridge = sp.stats.mstats.gmean(lmbda_array_Ridge, axis=1)
            
            #calculate the model using the Ridge method
            I = np.eye( np.shape(Phi_train)[1] )
            Ridgebeta = np.linalg.pinv( Phi_train.T @ Phi_train + lmbda_array_Ridge[i,j]*I ) @Phi_train.T @ z_train
            
            #calculate the optimal lambda value using scikit-learn GridSearchCV.
            #had to decrease the tolerance for LASSO because some test didn't converge.
            #this is also the part that causes the code to use a long time to run.
            #it would be a massive speedup if Ridge and Lasso had the same optimal lambda.
            modelLasso = Lasso(tol=0.0001, fit_intercept=False)
            searchLasso = GridSearchCV(modelLasso, parameters, scoring='neg_mean_squared_error', cv=kf)
            searchLasso.fit(Phi_train, z_train)
            lmbda_array_Lasso[i,j] = searchLasso.best_params_['alpha']
            avg_lmbda_Lasso = sp.stats.mstats.gmean(lmbda_array_Lasso, axis=1)
            
            #calculate the model using the LASSO method
            LassoReg = Lasso(alpha=lmbda_array_Lasso[i,j], fit_intercept=False) # Not include intercept
            LassoReg.fit(Phi_train, z_train)
            
            z_train = stdsc_z.inverse_transform(z_train)
            z_test = stdsc_z.inverse_transform(z_test)
            
            #rescale the prediction & model
            z_pred_OLS    = stdsc_z.inverse_transform( (Phi_test @ OLSbeta).reshape(-1,1)    )
            z_tilde_OLS   = stdsc_z.inverse_transform( (Phi_train @ OLSbeta).reshape(-1,1)   )
            z_pred_Ridge  = stdsc_z.inverse_transform( (Phi_test @ Ridgebeta).reshape(-1,1)  )
            z_tilde_Ridge = stdsc_z.inverse_transform( (Phi_train @ Ridgebeta).reshape(-1,1) )
            #z_Lasso and z_Ridge should have the same shape
            z_pred_Lasso  = stdsc_z.inverse_transform( LassoReg.predict(Phi_test).reshape(z_pred_Ridge.shape)   )
            z_tilde_Lasso = stdsc_z.inverse_transform( LassoReg.predict(Phi_train).reshape(z_tilde_Ridge.shape) )


            mse_pred_OLS[j]    = np.mean( (z_test - z_pred_OLS)**2     )
            mse_tilde_OLS[j]   = np.mean( (z_train - z_tilde_OLS)**2   )
            mse_pred_Ridge[j]  = np.mean( (z_test - z_pred_Ridge)**2   )
            mse_tilde_Ridge[j] = np.mean( (z_train - z_tilde_Ridge)**2 )
            mse_pred_Lasso[j]  = np.mean( (z_test - z_pred_Lasso)**2   )
            mse_tilde_Lasso[j] = np.mean( (z_train - z_tilde_Lasso)**2 )
            
            
        
        error_test[i, 0]  = np.mean( mse_pred_OLS    )
        error_test[i, 1]  = np.mean( mse_pred_Ridge  )
        error_test[i, 2]  = np.mean( mse_pred_Lasso  )
        error_train[i, 0] = np.mean( mse_tilde_OLS   )
        error_train[i, 1] = np.mean( mse_tilde_Ridge )
        error_train[i, 2] = np.mean( mse_tilde_Lasso )
        
        
        
    title_list = ['OLS Regression', 
                  'Ridge regression using optimal $\lambda$ at every complexity', 
                  'LASSO regression using optimal $\lambda$ at every complexity']
        
    fig1 = plt.figure(figsize=(15,5), dpi=200)
    ax1 = fig1.subplots(1,3)
    fig1.suptitle('Mean Square Error at different complexities')
    for i in range(3):   
        ax1[i].set_yscale('log')
        ax1[i].plot(deg, error_test[:, i], label='Testing MSE', color='red')
        ax1[i].plot(deg, error_train[:, i], label='Training MSE', color='blue')
        ax1[i].set_xlabel('Polynomial degree')
        ax1[i].set_ylabel('Mean Square Error')
        ax1[i].set_xticks(deg)
        ax1[i].set_title(title_list[i], fontsize=10)
        ax1[i].legend()
        ax1[i].grid()
        
    plt.show()
    
    
    plt.figure(dpi=200)
    plt.yscale('log')
    plt.plot(deg, avg_lmbda_Ridge, label='Optimal $\lambda$ value (Ridge)', linestyle='dashed', color='black')
    plt.plot(deg, avg_lmbda_Lasso, label='Optimal $\lambda$ value (LASSO)', linestyle='dashdot', color='black')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Geometric mean of $\lambda$')
    plt.xticks(deg)
    plt.yticks(alpha_array)
    plt.ylim(10**(a-1/2), 10**(b+1/2))
    plt.title('Geometric mean of optimal $\lambda$ for each complexity')
    plt.grid()
    plt.legend()
    plt.show()



def crossvalidation_NoLasso(x, y, z, mindeg=1, maxdeg=10, interval=1, k=10, a=-6, b=1):
    #Do k-fold cross validation to find the optimal complexity for the model
    #For each complexity, when using Ridge regression, the optimal lambda must be found

    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    X = np.vstack((x_flat,y_flat)).T
    # Note: X[:,0] = x_flat  X[:,1] = y_flat
    
    deg = np.arange(mindeg, maxdeg+interval, interval)
    
    #to use GridSearchCV you need a dictionary with parameters
    #appearantly lambda is called alpha in the scikit learn methods
    #according to the documentation the dictionary needs a list and not an array
    alpha_array = np.logspace(a, b, (b-a)+1)
    alpha_list = alpha_array.tolist()
    parameters = {'alpha': alpha_list}
    
    error_test    = np.zeros( (len(deg),2) )
    error_train    = np.zeros( (len(deg),2) )
    
    lmbda_array_Ridge = np.zeros( (len(deg),k) )
    lmbda_array_Lasso = np.zeros( (len(deg),k) )
    
    mse_pred_OLS    = np.zeros(k)
    mse_tilde_OLS   = np.zeros(k)
    mse_pred_Ridge  = np.zeros(k)
    mse_tilde_Ridge = np.zeros(k)
    mse_pred_Lasso  = np.zeros(k)
    mse_tilde_Lasso = np.zeros(k)
    
    
    for i in tqdm( range(len(deg)) ):
        #random_state=1 is for reproducibility, remove the keyword for random 
        #shuffle each time.
        kf = KFold(n_splits=k, shuffle=True)
    
    
        for j, (train, test) in enumerate(kf.split(X)):
            #make the testing and training data from k-fold
            X_train, X_test, z_train, z_test = X[train], X[test], z_flat[train], z_flat[test]
            
            stdsc_z = StandardScaler()
            z_train = stdsc_z.fit_transform( z_train.reshape(-1,1) )
            z_test  = stdsc_z.fit_transform( z_test.reshape(-1,1)  )
            
            #Make the design matrices
            Phi_train = Design_Matrix_2D(deg[i], X_train)
            Phi_test = Design_Matrix_2D(deg[i], X_test)
            
            #scale the design matrices
            stdsc = StandardScaler() # For x- and y-vals
            Phi_train = stdsc.fit_transform(Phi_train)
            Phi_test = stdsc.transform(Phi_test)
            
            
            #calculate the model using the OLS method
            OLSbeta = np.linalg.pinv(Phi_train.T @ Phi_train) @ Phi_train.T @ z_train
            
            #calculate the optimal lambda value using scikit-learn GridSearchCV
            modelRidge = Ridge(fit_intercept=False)
            searchRidge = GridSearchCV(modelRidge, parameters, scoring='neg_mean_squared_error', cv=kf)
            searchRidge.fit(Phi_train, z_train)
            lmbda_array_Ridge[i,j] = searchRidge.best_params_['alpha']
            avg_lmbda_Ridge = sp.stats.mstats.gmean(lmbda_array_Ridge, axis=1)
            
            #calculate the model using the Ridge method
            I = np.eye( np.shape(Phi_train)[1] )
            Ridgebeta = np.linalg.pinv( Phi_train.T @ Phi_train + lmbda_array_Ridge[i,j]*I ) @Phi_train.T @ z_train
            
            
            z_train = stdsc_z.inverse_transform(z_train)
            z_test = stdsc_z.inverse_transform(z_test)
            
            #rescale the prediction & model
            z_pred_OLS    = stdsc_z.inverse_transform( (Phi_test @ OLSbeta).reshape(-1,1)    )
            z_tilde_OLS   = stdsc_z.inverse_transform( (Phi_train @ OLSbeta).reshape(-1,1)   )
            z_pred_Ridge  = stdsc_z.inverse_transform( (Phi_test @ Ridgebeta).reshape(-1,1)  )
            z_tilde_Ridge = stdsc_z.inverse_transform( (Phi_train @ Ridgebeta).reshape(-1,1) )


            mse_pred_OLS[j]    = np.mean( (z_test - z_pred_OLS)**2     )
            mse_tilde_OLS[j]   = np.mean( (z_train - z_tilde_OLS)**2   )
            mse_pred_Ridge[j]  = np.mean( (z_test - z_pred_Ridge)**2   )
            mse_tilde_Ridge[j] = np.mean( (z_train - z_tilde_Ridge)**2 )
            
            
        
        error_test[i, 0]  = np.mean( mse_pred_OLS    )
        error_test[i, 1]  = np.mean( mse_pred_Ridge  )
        error_train[i, 0] = np.mean( mse_tilde_OLS   )
        error_train[i, 1] = np.mean( mse_tilde_Ridge )
        
        
        
    title_list = ['OLS Regression', 
                  'Ridge regression using optimal $\lambda$ at every complexity']
        
    fig1 = plt.figure(figsize=(15,5), dpi=200)
    ax1 = fig1.subplots(1,2)
    fig1.suptitle('Mean Square Error at different complexities')
    for i in range(2):   
        ax1[i].set_yscale('log')
        ax1[i].plot(deg, error_test[:, i], label='Testing MSE', color='red')
        ax1[i].plot(deg, error_train[:, i], label='Training MSE', color='blue')
        ax1[i].set_xlabel('Polynomial degree')
        ax1[i].set_ylabel('Mean Square Error')
        ax1[i].set_xticks(deg)
        ax1[i].set_title(title_list[i], fontsize=10)
        ax1[i].legend()
        ax1[i].grid()
        
    plt.show()
    
    
    plt.figure(dpi=200)
    plt.yscale('log')
    plt.plot(deg, avg_lmbda_Ridge, label='Optimal $\lambda$ value (Ridge)', linestyle='dashed', color='black')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Geometric mean of $\lambda$')
    plt.xticks(deg)
    plt.yticks(alpha_array)
    plt.ylim(10**(a-1/2), 10**(b+1/2))
    plt.title('Geometric mean of optimal $\lambda$ for each complexity')
    plt.grid()
    plt.legend()
    plt.show()





if __name__ == "__main__":
    #----------------------------------- Making data -------------------------------------------------------------------------------
    special_n = 20+1
    x = np.linspace(0, 1, special_n)
    y = np.linspace(0, 1, special_n)
    x, y = np.meshgrid(x,y)
    z = FrankeFunction(x, y)
    
    #comment and uncomment to actually run parts of the code
    #regression()
    crossvalidation(x, y, z)