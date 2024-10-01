import numpy as np
import matplotlib.pyplot as plt
from random import random, seed
from tqdm import tqdm

from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from bg_taskabc import MSE, R2, FrankeFunction, Design_Matrix_2D



"""
This code file (op_task_e.py) is supposed to replace op_task-e.py because you can import functions
from op_task_e.py. Python doesn't like filenames with hyphens in them.
The actual contents of this file is the same as op_task-e.py, only the name (and this
message) is supposed to be different.
If you see this message or/and op_task-e.py, they should have been deleted.
"""



np.random.seed(1) #set seed for easier troubleshooting



def regression(savefig=False):

    # ------------------------------- Make data -------------------------------------
    x = np.linspace(0, 1, 20+1)
    y = np.linspace(0, 1, 20+1)
    x, y = np.meshgrid(x,y)
    z = FrankeFunction(x, y)

    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    X = np.vstack((x_flat,y_flat)).T
    X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.25)
    # Note: X[:,0] = x_flat  X[:,1] = y_flat

    deg = np.arange(1,6) #degrees of polynomial
    OLSbeta_list = [] 
    Ridgebeta_list = []
    Lassobeta_list = []
    OLS_MSE = []
    OLS_R2 = []
    Ridge_MSE = []
    Ridge_R2 = []
    Lasso_MSE = []
    Lasso_R2 = []


    fig               = plt.figure(figsize=(12,7))
    fig_Ridge         = plt.figure(figsize=(12,7))
    fig_Lasso         = plt.figure(figsize=(12,7))
    
    
    
    for i in range(len(deg)):
        #----------------------------------------   Scaling ------------------------------------------
        stdsc = StandardScaler() # For x- and y-vals
        X_train = stdsc.fit_transform(X_train)
        X_test = stdsc.transform(X_test)

        stdsc_z = StandardScaler() # For z-vals
        z_train = stdsc_z.fit_transform(z_train.reshape(-1,1))
        z_test = stdsc_z.transform(z_test.reshape(-1,1))
        # reshape to get 2D array
        # .transform() expect 2D array -> [values,] -> [values, 1]


        # Making Design Matrix Phi 
        Phi_train = Design_Matrix_2D(deg[i], X_train)
        Phi_test = Design_Matrix_2D(deg[i], X_test)



        #-----------------------------------------   OLS - Regression    -----------------------------
        OLSbeta = np.linalg.inv(Phi_train.T @ Phi_train) @ Phi_train.T @ z_train
        z_pred = Phi_test @ OLSbeta
        z_tilde = Phi_train @ OLSbeta

        # Adding OLSbeta to list 
        OLSbeta_list.append(OLSbeta)
        
    

        # ---------------------------------- MANUALLY RIDGE + SCIKIT-LEARN LASSO ----------------------
        num_terms = int((deg[i] + 1) * (deg[i] + 2) / 2 - 1) # From Design_Matrix_2D()
        I = np.eye(num_terms, num_terms)

        lambdas = [0.0001,0.001,0.01,0.1,1]

        z_tilde_Ridge = np.zeros((len(lambdas), X_train.shape[0],1))
        z_pred_Ridge = np.zeros((len(lambdas), X_test.shape[0], 1))

        z_tilde_Lasso = np.zeros((len(lambdas), X_train.shape[0], 1))
        z_pred_Lasso = np.zeros((len(lambdas), X_test.shape[0], 1))

        for j in range(len(lambdas)):
            lmb = lambdas[j]

            Ridgebeta = np.linalg.inv(Phi_train.T @ Phi_train + lmb*I) @ Phi_train.T @ z_train

            LassoReg = Lasso(lmb, fit_intercept=False) # Not include intercept
            LassoReg.fit(Phi_train, z_train)
            
            # and then make the prediction
            z_tilde_Ridge[j] = Phi_train @ Ridgebeta
            z_pred_Ridge[j] = Phi_test @ Ridgebeta

            z_tilde_Lasso[j] = LassoReg.predict(Phi_train).reshape(z_tilde_Lasso[j].shape)
            z_pred_Lasso[j] = LassoReg.predict(Phi_test).reshape(z_pred_Lasso[j].shape)

            # Adding beta's to list 
            Ridgebeta_list.append(Ridgebeta)
            Lassobeta_list.append(LassoReg.coef_)


        # -------------------------------------------- Rescaling --------------------------------------
        #Reverse wScaling with StandardScaler.inverse_transform()
        X_test = stdsc.inverse_transform(X_test)
        x_test = X_test[:,0]
        y_test = X_test[:,1]

        X_train = stdsc.inverse_transform(X_train)
        x_train = X_train[:,0]
        y_train = X_train[:,1]
        
        z_test  = stdsc_z.inverse_transform(z_test)
        z_train = stdsc_z.inverse_transform(z_train)

        # OLS--------------
        z_pred  = stdsc_z.inverse_transform(z_pred)
        z_tilde = stdsc_z.inverse_transform(z_tilde) 
        
        # Ridge------------
        z_pred_Ridge  = [ stdsc_z.inverse_transform(z_pred_Ridge[j])  for j in range(len(lambdas)) ]
        z_tilde_Ridge = [ stdsc_z.inverse_transform(z_tilde_Ridge[j]) for j in range(len(lambdas)) ]
        
        # Lasso-------------
        z_pred_Lasso  = [ stdsc_z.inverse_transform(z_pred_Lasso[j])  for j in range(len(lambdas)) ]
        z_tilde_Lasso = [ stdsc_z.inverse_transform(z_tilde_Lasso[j]) for j in range(len(lambdas)) ]
    


        # ------------------------------------------ MSE -----------------------------------------
        for j in range(len(lambdas)):
            # Ridge
            Ridge_MSE.append( [ MSE(z_train, z_tilde_Ridge[j]), MSE(z_test, z_pred_Ridge[j]) ] )
            Ridge_R2.append(  [ R2(z_train, z_tilde_Ridge[j]) , R2(z_test, z_pred_Ridge[j])  ] )

            # Lasso
            Lasso_MSE.append( [ MSE(z_train, z_tilde_Lasso[j]), MSE(z_test, z_pred_Lasso[j]) ] )
            Lasso_R2.append(  [ R2(z_train, z_tilde_Lasso[j]) , R2(z_test, z_pred_Lasso[j])  ] )


        OLS_MSE.append( [MSE(z_train, z_tilde), MSE(z_test, z_pred) ] )
        OLS_R2.append(  [R2(z_train, z_tilde) , R2(z_test, z_pred)  ] )



        # ------------------------------------- Plotting -------------------------------------
        
        # OLS
        ax = fig.add_subplot(2, 3, i+1, projection='3d')

        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False, alpha = 0.7)

        pred = ax.scatter(x_test, y_test, z_pred, color='r', s=10, label='z_Pred')
        tilde = ax.scatter(x_train, y_train, z_tilde, color='g', s=10, label='z_Tilde')

        fig.suptitle('OLS Regression', fontsize=16)
        ax.set_title(f'Polynomial Degree {deg[i]}', fontsize=10)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_zlim(-0.10, 1.40)
        ax.legend(loc='upper left', fontsize='small')


        
        # Ridge
        axR = fig_Ridge.add_subplot(2, 3, i+1, projection='3d')

        surf = axR.plot_surface(x, y, z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False, alpha = 0.7)
        # Plot only for highest lambda.
        # Chaos to spot differences by plot.
        # Will look at MSE and R2
        pred = axR.scatter(x_test, y_test, z_pred_Ridge[-1], color='r', s=10, label=f'z_Pred, lmd = {lambdas[-1]}') 
        tilde = axR.scatter(x_train, y_train, z_tilde_Ridge[-1], color='g', s=10, label=f'z_Tilde, lmd = {lambdas[-1]}')

        fig_Ridge.suptitle(f'Ridge Regression\n  Lambda = {lambdas[-1]}', fontsize=16)
        axR.set_title(f'Polynomial Degree {deg[i]}', fontsize=10)
        axR.set_xlabel('X axis')
        axR.set_ylabel('Y axis')
        axR.set_zlabel('Z axis')
        axR.set_zlim(-0.10, 1.40)
        axR.legend(loc='upper left', fontsize='small')
        


        # Lasso
        axL = fig_Lasso.add_subplot(2, 3, i+1, projection='3d')

        surf = axL.plot_surface(x, y, z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False, alpha = 0.7)
        # Plot only for one lambda.
        # Chaos to spot differences by plot.
        # Will look at MSE and R2

        test = False
        if test:
            # Test Lasso lambda = 1
            # See functionality of Lasso
            pred = axL.scatter(x_test, y_test, z_pred_Lasso[-1], color='r', s=10, label=f'z_Pred, lmd = {lambdas[-1]}') 
            tilde = axL.scatter(x_train, y_train, z_tilde_Lasso[-1], color='g', s=10, label=f'z_Tilde, lmd = {lambdas[-1]}')
            fig_Lasso.suptitle(f'TEST: Lasso Regression\n  Lambda = {lambdas[-1]}', fontsize=16)
        else: 
            k = -3 # Choose which lambda to plot
            pred = axL.scatter(x_test, y_test, z_pred_Lasso[k], color='r', s=10, label=f'z_Pred, lmd = {lambdas[k]}') 
            tilde = axL.scatter(x_train, y_train, z_tilde_Lasso[k], color='g', s=10, label=f'z_Tilde, lmd = {lambdas[k]}')
            fig_Lasso.suptitle(f'Lasso Regression\n  Only lambda = {lambdas[k]}', fontsize=16)

        axL.set_title(f'Polynomial Degree {deg[i]}', fontsize=10)
        axL.set_xlabel('X axis')
        axL.set_ylabel('Y axis')
        axL.set_zlabel('Z axis')
        axL.set_zlim(-0.10, 1.40)
        axL.legend(loc='upper left', fontsize='small')
    

    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    if savefig == True:
        fig.savefig('OLS.png')
        fig_Ridge.savefig('Ridge.png')
    plt.show()



    # ------------------------------------------------ PRINTING INFO ------------------------------------------------------
    want_beta = False #True # if want to see beta-values


    # OLS DATA
    OLS_MSE = np.array(OLS_MSE)
    OLS_R2 = np.array(OLS_R2)
    print('--------------------------------- OLS - Regression')
    print('Elements: Beta Transpose, MSE, R2')
    for d, b, MSE_train, MSE_test, R2_train, R2_test in zip(deg, OLSbeta_list, OLS_MSE[:,0], OLS_MSE[:,1], OLS_R2[:,0], OLS_R2[:,1]):
        print(f'\nPolynomial Degree  {d}') 
        print(f'MSE_train = {MSE_train:10.3e}    MSE_test = {MSE_test:10.3e}    R2_train  = {R2_train:10.3e}    R2_test  = {R2_test:10.3e}')
        if want_beta:
            print(f'Beta = {b.T}')


    #print some spacing in between all the numbers
    print()
    print()


    # Ridge DATA
    print('--------------------------------- Ridge - Regression')
    print('Elements: Beta Transpose, MSE, R2')
    for i in range(len(deg)):
        print(f'\nPolynomial Degree  {deg[i]}') 
        #print(Ridgebeta_list)
        for j in range(len(lambdas)):
            print(
                f'Lambda = {lambdas[j]:10}:    MSE_train = {Ridge_MSE[i*len(lambdas) + j][0]:10.3e}    MSE_test = {Ridge_MSE[i*len(lambdas) + j][1]:10.3e}     '
                f'R2_train  = {Ridge_R2[i*len(lambdas) + j][0]:10.3e}    R2_test  = {Ridge_R2[i*len(lambdas) + j][1]:10.3e}'
                )
            if want_beta:
                print(f'Beta = {Ridgebeta_list[i*len(lambdas) + j].T} \n')
    
    
    #print some spacing in between all the numbers
    print()
    print()


    # Lasso DATA
    print('--------------------------------- Lasso - Regression')
    print('Elements: Beta Transpose, MSE, R2')
    for i in range(len(deg)):
        print(f'\nPolynomial Degree  {deg[i]}') 
        #print(Ridgebeta_list)
        for j in range(len(lambdas)):
            print(
                f'Lambda = {lambdas[j]:10}:    MSE_train = {Lasso_MSE[i*len(lambdas) + j][0]:10.3e}    MSE_test = {Lasso_MSE[i*len(lambdas) + j][1]:10.3e}     '
                f'R2_train  = {Lasso_R2[i*len(lambdas) + j][0]:10.3e}    R2_test  = {Lasso_R2[i*len(lambdas) + j][1]:10.3e}'
                )
            if want_beta:
                print(f'Beta = {Lassobeta_list[i*len(lambdas) + j].T} \n')



def bootstrap_num(data=1, use_real_data=False, n_bootstraps=100, row_start=1000, special_deg=5, min_n=10+1, max_n=80+1, interval=5):
    #I usually add one to the number of points to make the points in x and y
    #have a simpler decimal representation.
    #Example: linspace(0,1,10) = (0, 0.111, 0.222, 0.333, 0.444, 0.555, 0.666, 0.777, 0.888, 1)
    
    #max_n+1 because you want to include the actual value of max_n
    n_list = np.arange(min_n, max_n+min_n+1, interval)
    
    #declare all the useful arrays to store the calculated values
    error_test    = np.zeros(len(n_list))
    bias_test     = np.zeros(len(n_list))
    variance_test = np.zeros(len(n_list))
    
    error_train    = np.zeros(len(n_list))
    bias_train     = np.zeros(len(n_list))
    variance_train = np.zeros(len(n_list))
    
    if use_real_data == True:
        #only have to actually read the data if you want to use it
        surf = imread(data)
    
    for i in tqdm( range(len(n_list)) ):
        
        if use_real_data == False:
            #make data using FrankeFunction
            x      = np.linspace(0, 1, int(n_list[i]))
            y      = np.linspace(0, 1, int(n_list[i]))
            x, y   = np.meshgrid(x, y)
            z      = FrankeFunction(x, y)
        
        else:
            #turn the data into a usable format and declare x and y
            row_end = row_start + int(n_list[i])     # End of rows 

            col_start = row_start   # Start of rows 
            col_end = row_end       # End of rows 

            # Extract the center half of the matrix
            surf = surf[row_start:row_end, col_start:col_end]
            
            z = surf

            m = len(surf)
            x = np.linspace(0,1,m)
            y = np.linspace(0,1,m)
            x, y = np.meshgrid(x,y)

        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()

        X = np.vstack((x_flat,y_flat)).T
        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.25)
        
        z_train = z_train.reshape(-1,1)
        z_test = z_test.reshape(-1,1)
    
        #compute the Phi matrices from unscrambled/"bootstrap-ed" data
        Phi_test  = Design_Matrix_2D(special_deg, X_test)
        Phi_train = Design_Matrix_2D(special_deg, X_train)
        
        #scale the design matrices
        stdsc = StandardScaler() # For x- and y-vals
        Phi_train = stdsc.fit_transform(Phi_train)
        Phi_test = stdsc.transform(Phi_test)
        
        #initialise different arrays to store the different models/predictions from bootstrap
        z_pred  = np.zeros( (X_test.shape[0], n_bootstraps)  )
        z_tilde = np.zeros( (X_train.shape[0], n_bootstraps) )
        
        #loop to do the bootstrap technique a certain amount of times
        for j in range(n_bootstraps):
            #scramble the scaled data
            X_, z_ = resample(X_train, z_train)
            
            stdsc_z_ = StandardScaler()
            z_ = stdsc_z_.fit_transform(z_.reshape(-1,1))
            
            #evaluate the new model on the same testing and training data each time.
            Phi_train_ = Design_Matrix_2D(special_deg, X_)
            stdsc_ = StandardScaler()
            Phi_train_ = stdsc_.fit_transform(Phi_train_)
            
            OLSbeta_ = np.linalg.inv(Phi_train_.T @ Phi_train_) @ Phi_train_.T @ z_
            
            z_pred[:, j]  = stdsc_z_.inverse_transform( (Phi_test @ OLSbeta_).reshape(1,-1)  )
            z_tilde[:, j] = stdsc_z_.inverse_transform( (Phi_train @ OLSbeta_).reshape(1,-1) )

        
        error_test[i]     = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
        bias_test[i]      = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
        variance_test[i]  = np.mean( np.var(z_pred, axis=1, keepdims=True)                )
        
        error_train[i]    = np.mean( np.mean((z_train - z_tilde)**2, axis=1, keepdims=True) )
        bias_train[i]     = np.mean( (z_train - np.mean(z_tilde, axis=1, keepdims=True))**2 )
        variance_train[i] = np.mean( np.var(z_tilde, axis=1, keepdims=True)                 )
    

    
    plt.figure(dpi=200)
    plt.yscale('log')
    plt.plot(n_list, error_test, label='Mean Square Error', color='red')
    plt.plot(n_list, bias_test, label='Bias', color='blue')
    plt.plot(n_list, variance_test, label='Variance', color='lime')
    plt.xlabel('Number of datapoints (in x & y direction)')
    plt.ylabel('Error')
    plt.xticks(n_list)
    plt.title('Bias-Variance Trade Off (Number of datapoints)', fontsize=10)
    plt.legend()
    plt.grid()
    plt.show()
    


def bootstrap_comp(x, y, z, n_bootstraps=100, mindeg=1, maxdeg=10, interval=1):

    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    X = np.vstack((x_flat,y_flat)).T
    X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.25)
    # Note: X[:,0] = x_flat  X[:,1] = y_flat
    
    deg = np.arange(mindeg, maxdeg+1, interval) #degrees of polynomial
    
    
    error_test    = np.zeros(len(deg))
    bias_test     = np.zeros(len(deg))
    variance_test = np.zeros(len(deg))
    
    error_train = np.zeros(len(deg))
    bias_train = np.zeros(len(deg))
    variance_train = np.zeros(len(deg))
    
    for i in tqdm( range(len(deg)) ):
        z_train = z_train.reshape(-1,1)
        z_test = z_test.reshape(-1,1)
        
        #same number of points in the x and y direction
        z_pred = np.empty((X_test.shape[0], n_bootstraps))
        z_tilde = np.empty((X_train.shape[0], n_bootstraps))
        
        # Making Design Matrix Phi 
        Phi_train = Design_Matrix_2D(deg[i], X_train)
        Phi_test = Design_Matrix_2D(deg[i], X_test)
        
        #Scaling Design Matrix
        stdsc = StandardScaler() # For x- and y-vals
        Phi_train = stdsc.fit_transform(Phi_train)
        Phi_test = stdsc.transform(Phi_test)
        
        
        for j in range(n_bootstraps):
            X_, z_ = resample(X_train, z_train)
            
            #scale resampled z values
            stdsc_z_ = StandardScaler()
            z_ = stdsc_z_.fit_transform(z_.reshape(-1,1))
            
            #make and scale resampled design matrix
            Phi_train_ = Design_Matrix_2D(deg[i], X_)
            stdsc_ = StandardScaler()
            Phi_train_ = stdsc_.fit_transform(Phi_train_)
            
            OLSbeta_ = np.linalg.inv(Phi_train_.T @ Phi_train_) @ Phi_train_.T @ z_

            # Evaluate the new model on the same testing and training data each time.
            z_pred[:, j]  = stdsc_z_.inverse_transform( (Phi_test @ OLSbeta_).reshape(1,-1)  )
            z_tilde[:, j] = stdsc_z_.inverse_transform( (Phi_train @ OLSbeta_).reshape(1,-1) )
        
        
        error_test[i]    = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
        bias_test[i]     = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
        variance_test[i] = np.mean( np.var(z_pred, axis=1, keepdims=True)                )
        
        error_train[i]    = np.mean( np.mean((z_train - z_tilde)**2, axis=1, keepdims=True) )
        bias_train[i]     = np.mean( (z_train - np.mean(z_tilde, axis=1, keepdims=True))**2 )
        variance_train[i] = np.mean( np.var(z_tilde, axis=1, keepdims=True)                 )
    
    
    
    plt.figure(dpi=200)
    plt.yscale('log')
    plt.plot(deg, error_test, label='Mean Square Error', color='red')
    plt.plot(deg, bias_test, label='Bias', color='blue')
    plt.plot(deg, variance_test, label='Variance', color='lime')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Error')
    plt.xticks(deg)
    plt.title('Bias-Variance Trade Off (Complexity)', fontsize=10)
    plt.legend()
    plt.grid()
    plt.show()





if __name__ == "__main__":
    #----------------------------------- Making data -------------------------------------------------------------------------------
    special_n=20+1
    x_comp = np.linspace(0, 1, special_n)
    y_comp = np.linspace(0, 1, special_n)
    x_comp, y_comp = np.meshgrid(x_comp,y_comp)
    z_comp = FrankeFunction(x_comp, y_comp)
    
    #comment and uncomment to actually run parts of the code
    #regression()
    bootstrap_num()
    bootstrap_comp(x_comp, y_comp, z_comp)