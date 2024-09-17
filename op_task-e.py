import numpy as np
import matplotlib.pyplot as plt
from random import random, seed

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample



"""
copied all the code from bg_taskabc.py (12.9.24), cleaned up a bit, and added resampling analysis.
"""



np.random.seed(1) #set seed for easier troubleshooting


"""
Support Functions
"""

def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def FrankeFunction(x,y):
        term1 = 0.75*np.exp( -(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2) )
        term2 = 0.75*np.exp( -((9*x+1)**2)/49.0 - 0.1*(9*y+1)       )
        term3 =  0.5*np.exp( -(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)    )
        term4 = -0.2*np.exp( -(9*x-4)**2 - (9*y-7)**2               )
        return term1 + term2 + term3 + term4


def Design_Matrix_2D(deg, X):
    # Design matrix for a function that takes in two input variables
    # The number of polynomial terms for two variables (x, y) up to degree d is (d+1)(d+2)/2
    # Minus 1 from dropping intercept-column
    num_terms = int((deg + 1) * (deg + 2) / 2 - 1)

    Phi = np.zeros((X.shape[0], num_terms))
    # PS: not include intercept in design matrix, will scale (centered values)
    #dx and dy = polynomial degree in x and y direction
    col = 0
    for dx in range(1, deg+1):
        for dy in range(dx+1):
            # X[:,0] = x-values
            # X[:,1] = y-values
            Phi[:,col] = ( X[:,0]**(dx - dy) ) * ( X[:,1]**dy )
            col += 1
    return Phi



"""
Computation Functions
"""

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



def bootstrap_num(n_bootstraps=100, special_deg=5, min_n=10+1, max_n=80+1, interval=5):
    #I usually add one to the number of points to make the points in x and y
    #have a simpler decimal representation.
    #Example: linspace(0,1,10) = (0, 0.111, 0.222, 0.333, 0.444, 0.555, 0.666, 0.777, 0.888, 1)
    
    n_list = np.arange(min_n, max_n+1, interval)
    
    error_test    = np.zeros(len(n_list))
    bias_test     = np.zeros(len(n_list))
    variance_test = np.zeros(len(n_list))
    
    error_train    = np.zeros(len(n_list))
    bias_train     = np.zeros(len(n_list))
    variance_train = np.zeros(len(n_list))
    
    for i in range(len(n_list)):
        #make data using FrankeFunction
        x      = np.linspace(0, 1, int(n_list[i]))
        y      = np.linspace(0, 1, int(n_list[i]))
        x, y   = np.meshgrid(x, y)
        z      = FrankeFunction(x, y)

        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()

        X = np.vstack((x_flat,y_flat)).T
        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.25)
        
        z_train = z_train.reshape(-1,1)
        z_test = z_test.reshape(-1,1)
        
        #scale the data
        stdsc = StandardScaler() # For x- and y-vals
        X_train_scaled = stdsc.fit_transform(X_train)
        X_test_scaled = stdsc.transform(X_test)
        
        #compute the Phi matrices from unscrambled/"bootstrap-ed" data
        Phi_test  = Design_Matrix_2D(special_deg, X_test_scaled)
        Phi_train = Design_Matrix_2D(special_deg, X_train_scaled)
        
        
        #initialise different arrays to store the different models/predictions from bootstrap
        z_pred  = np.empty( (X_test_scaled.shape[0], n_bootstraps)  )
        z_tilde = np.empty( (X_train_scaled.shape[0], n_bootstraps) )
        
        #loop to do the bootstrap technique a certain amount of times
        for j in range(n_bootstraps):
            #scramble the scaled data
            X_, z_ = resample(X_train, z_train)
            
            stdsc_ = StandardScaler()
            X_ = stdsc_.fit_transform(X_)
            
            stdsc_z_ = StandardScaler()
            z_ = stdsc_z_.fit_transform(z_.reshape(-1,1))
            
            #evaluate the new model on the same testing and training data each time.
            Phi_train_ = Design_Matrix_2D(special_deg, X_)
            OLSbeta_ = np.linalg.inv(Phi_train_.T @ Phi_train_) @ Phi_train_.T @ z_
            
            X_ = stdsc_.inverse_transform(X_)
            z_pred[:, j]  = stdsc_z_.inverse_transform( (Phi_test @ OLSbeta_).reshape(1,-1)  )
            z_tilde[:, j] = stdsc_z_.inverse_transform( (Phi_train @ OLSbeta_).reshape(1,-1) )

        
        error_test[i]    = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
        bias_test[i]     = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
        variance_test[i] = np.mean( np.var(z_pred, axis=1, keepdims=True)                )
        
        error_train[i]    = np.mean( np.mean((z_train - z_tilde)**2, axis=1, keepdims=True) )
        bias_train[i]     = np.mean( (z_train - np.mean(z_tilde, axis=1, keepdims=True))**2 )
        variance_train[i] = np.mean( np.var(z_tilde, axis=1, keepdims=True)                 )
    
    
    fig_BiasVariance  = plt.figure(figsize=(12,10))
    ax = fig_BiasVariance.subplots(2,1)
    fig_BiasVariance.suptitle('Bias-Variance Tradeoff (Bootstrap, # of datapoints)')
    
    #axBN[0].set_yscale("log")
    ax[0].plot(n_list, bias_test, label='Testing Bias', color='red')
    ax[0].plot(n_list, bias_train, label='Training Bias', color='blue')
    ax[0].set_xlabel('# of x (and y) points')
    ax[0].set_ylabel('Bias')
    ax[0].set_xticks(n_list)
    ax[0].set_title('Testing Bias vs Training Bias', fontsize=10)
    ax[0].legend()
    ax[0].grid()
    
    ax[1].set_yscale("log")
    ax[1].plot(n_list, variance_test, label='Testing Variance', color='red')
    ax[1].plot(n_list, variance_train, label='Training Variance', color='blue')
    ax[1].set_xlabel('# of x (and y) points')
    ax[1].set_ylabel('Variance')
    ax[1].set_xticks(n_list)
    ax[1].set_title('Testing Variance vs Training Variance', fontsize=10)
    ax[1].legend()
    ax[1].grid()
    plt.show()
    
    
    plt.figure(dpi=200)
    plt.plot(n_list, np.log10(error_test), label='Testing MSE', color='red')
    plt.plot(n_list, np.log(error_train), label='Training MSE', color='blue')
    plt.xlabel('# of x (and y) points')
    plt.ylabel('log(Mean Square Error)')
    plt.xticks(n_list)
    plt.title('Testing MSE vs Training MSE', fontsize=10)
    plt.legend()
    plt.grid()
    plt.show()
    



def bootstrap_comp(n_bootstraps=100, special_num=20+1, maxdeg=12):
    x = np.linspace(0, 1, special_num)
    y = np.linspace(0, 1, special_num)
    x, y = np.meshgrid(x,y)
    z = FrankeFunction(x, y)

    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    X = np.vstack((x_flat,y_flat)).T
    X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.25)
    # Note: X[:,0] = x_flat  X[:,1] = y_flat
    
    deg = np.arange(1, maxdeg+1) #degrees of polynomial
    
    
    error_test    = np.zeros(len(deg))
    bias_test     = np.zeros(len(deg))
    variance_test = np.zeros(len(deg))
    
    error_train = np.zeros(len(deg))
    bias_train = np.zeros(len(deg))
    variance_train = np.zeros(len(deg))
    
    for i in range(len(deg)):
        stdsc = StandardScaler() # For x- and y-vals
        X_train_scaled = stdsc.fit_transform(X_train)
        X_test_scaled = stdsc.transform(X_test)

        z_train = z_train.reshape(-1,1)
        z_test = z_test.reshape(-1,1)
        
        #same number of points in the x and y direction
        z_pred = np.empty((X_test.shape[0], n_bootstraps))
        z_tilde = np.empty((X_train.shape[0], n_bootstraps))
        
        # Making Design Matrix Phi 
        Phi_train = Design_Matrix_2D(deg[i], X_train_scaled)
        Phi_test = Design_Matrix_2D(deg[i], X_test_scaled)
        
        
        for j in range(n_bootstraps):
            X_, z_ = resample(X_train, z_train)
            
            stdsc_ = StandardScaler()
            X_ = stdsc_.fit_transform(X_)
            
            stdsc_z_ = StandardScaler()
            z_ = stdsc_z_.fit_transform(z_.reshape(-1,1))
            
            # Evaluate the new model on the same testing and training data each time.
            Phi_train_ = Design_Matrix_2D(deg[i], X_)
            OLSbeta_ = np.linalg.inv(Phi_train_.T @ Phi_train_) @ Phi_train_.T @ z_

            X_ = stdsc_.inverse_transform(X_)
            z_pred[:, j]  = stdsc_z_.inverse_transform( (Phi_test @ OLSbeta_).reshape(1,-1)  )
            z_tilde[:, j] = stdsc_z_.inverse_transform( (Phi_train @ OLSbeta_).reshape(1,-1) )

        
        error_test[i]    = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
        bias_test[i]     = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
        variance_test[i] = np.mean( np.var(z_pred, axis=1, keepdims=True)                )
        
        error_train[i]    = np.mean( np.mean((z_train - z_tilde)**2, axis=1, keepdims=True) )
        bias_train[i]     = np.mean( (z_train - np.mean(z_tilde, axis=1, keepdims=True))**2 )
        variance_train[i] = np.mean( np.var(z_tilde, axis=1, keepdims=True)                 )
    
    
    fig_BootstrapComp = plt.figure(figsize=(12,10))
    
    axBC = fig_BootstrapComp.subplots(2,1)
    fig_BootstrapComp.suptitle('Bias-Variance Tradeoff (Bootstrap, Complexity)')
    
    axBC[0].plot(deg, bias_test, label='Testing Bias', color='red')
    axBC[0].plot(deg, bias_train, label='Training Bias', color='blue')
    axBC[0].set_xlabel('Polynomial degree')
    axBC[0].set_ylabel('Bias')
    axBC[0].set_xticks(deg)
    axBC[0].set_title('Testing Bias vs Training Bias', fontsize=10)
    axBC[0].legend()
    axBC[0].grid()
    
    axBC[1].plot(deg, variance_test, label='Testing Variance', color='red')
    axBC[1].plot(deg, variance_train, label='Training Variance', color='blue')
    axBC[1].set_xlabel('Polynomial degree')
    axBC[1].set_ylabel('Variance')
    axBC[1].set_xticks(deg)
    axBC[1].set_title('Testing Variance vs Training Variance', fontsize=10)
    axBC[1].legend()
    axBC[1].grid()
    
    plt.show()
    
    
    plt.figure(dpi=200)
    plt.plot(deg, np.log10(error_test), label='Testing MSE', color='red')
    plt.plot(deg, np.log10(error_train), label='Training MSE', color='blue')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('log(Mean Square Error)')
    plt.xticks(deg)
    plt.title('Testing MSE vs Training MSE', fontsize=10)
    plt.legend()
    plt.grid()
    plt.show()



#comment and uncomment to actually run parts of the code
#regression()
bootstrap_num()
bootstrap_comp()