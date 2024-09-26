import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

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


def regression(x, y, z, deg_max = 6, bool_info = False):

    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Create new matrix X for train / test -splitting  
    # only one imput for x (both x and y) and one for z
    X = np.vstack((x_flat,y_flat)).T
    # Note: X[:,0] = x-vals  X[:,1] = y-vals

    X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.25)

    if __name__ == "__main__": # if run bg_taskabc.py
        deg = np.arange(deg_max - 5, deg_max) # degrees of polynomial
    else:                      # if run task_g.py
        deg = np.arange(deg_max - 25, deg_max+1,5) # degrees of polynomial

    OLSbeta_list   = [] 
    Ridgebeta_list = []
    Lassobeta_list = []
    OLS_MSE        = []
    OLS_R2         = []
    Ridge_MSE      = []
    Ridge_R2       = []
    Lasso_MSE      = []
    Lasso_R2       = []


    if __name__ == "__main__":  # if run bg_taskabc.py
        fig       = plt.figure(figsize=(12,7))
        fig_Ridge = plt.figure(figsize=(12,7))
        fig_Lasso = plt.figure(figsize=(12,7))
        fig_Lasso_Appendix = plt.figure(figsize=(12,7))
        
    else:                        # if run task_g.py
        fig       = plt.figure(figsize=(7,7))
        fig_Ridge = plt.figure(figsize=(7,7))
        fig_Lasso = plt.figure(figsize=(7,7))
    
    
    for i in range(len(deg)):
        #----------------------------------------   Scaling -----------------------------
        stdsc = StandardScaler() # For x- and y-vals
        X_train = stdsc.fit_transform(X_train)
        X_test = stdsc.transform(X_test)

        stdsc_z = StandardScaler() # For z-vals
        z_train = stdsc_z.fit_transform(z_train.reshape(-1,1))
        z_test = stdsc_z.transform(z_test.reshape(-1,1))
        # reshape to get 2D array
        # function .fit_transform() and .transform() expect 2D array 


        # Making Design Matrix Phi 
        Phi_train = Design_Matrix_2D(deg[i], X_train)
        Phi_test  = Design_Matrix_2D(deg[i], X_test)



        #-----------------------------------------   OLS - Regression    -----------------------------
        OLSbeta = np.linalg.inv(Phi_train.T @ Phi_train) @ Phi_train.T @ z_train
        z_pred = Phi_test @ OLSbeta
        z_tilde = Phi_train @ OLSbeta

        # Adding OLSbeta to list 
        OLSbeta_list.append(OLSbeta)


        # ---------------------------------- MANUALLY RIDGE + SCIKIT-LEARN LASSO ----------------------
        num_terms = int((deg[i] + 1) * (deg[i] + 2) / 2 - 1) # Numbers from Design_Matrix_2D()
        I = np.eye(num_terms,num_terms)

        lambdas = [0.0001,0.001,0.01,0.1,1]

        z_tilde_Ridge = np.zeros((len(lambdas), X_train.shape[0],1))
        z_pred_Ridge = np.zeros((len(lambdas), X_test.shape[0], 1))

        z_tilde_Lasso = np.zeros((len(lambdas), X_train.shape[0], 1))
        z_pred_Lasso = np.zeros((len(lambdas), X_test.shape[0], 1))

        for j in range(len(lambdas)):
            lmb = lambdas[j]

            Ridgebeta = np.linalg.inv(Phi_train.T @ Phi_train + lmb*I) @ Phi_train.T @ z_train

            # Suppress ConvergenceWarning
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            LassoReg = Lasso(lmb, fit_intercept=False) # Not include intercept + setting tol to not get error-messages
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
        x_test = X_test[:,0].reshape(-1,1)
        y_test = X_test[:,1].reshape(-1,1)

        X_train = stdsc.inverse_transform(X_train)
        x_train = X_train[:,0].reshape(-1,1)
        y_train = X_train[:,1].reshape(-1,1)
        
        z_test = stdsc_z.inverse_transform(z_test)
        z_train = stdsc_z.inverse_transform(z_train)

        # OLS--------------
        z_pred = stdsc_z.inverse_transform(z_pred)
        z_tilde = stdsc_z.inverse_transform(z_tilde) 
        
        # Ridge------------
        z_pred_Ridge = [stdsc_z.inverse_transform(z_pred_Ridge[j]) for j in range(len(lambdas))]
        z_tilde_Ridge = [stdsc_z.inverse_transform(z_tilde_Ridge[j]) for j in range(len(lambdas))]
        
        # Lasso-------------
        z_pred_Lasso = [stdsc_z.inverse_transform(z_pred_Lasso[j]) for j in range(len(lambdas))]
        z_tilde_Lasso = [stdsc_z.inverse_transform(z_tilde_Lasso[j]) for j in range(len(lambdas))]
        
    

        # ------------------------------------------ MSE -----------------------------------------
        for j in range(len(lambdas)):
            # Ridge
            Ridge_MSE.append([MSE(z_train, z_tilde_Ridge[j]),MSE(z_test, z_pred_Ridge[j])])
            Ridge_R2.append([R2(z_train, z_tilde_Ridge[j]), R2(z_test, z_pred_Ridge[j])])

            # Lasso
            Lasso_MSE.append([MSE(z_train, z_tilde_Lasso[j]),MSE(z_test, z_pred_Lasso[j])])
            Lasso_R2.append([R2(z_train, z_tilde_Lasso[j]), R2(z_test, z_pred_Lasso[j])])
    

        OLS_MSE.append([MSE(z_train, z_tilde), MSE(z_test, z_pred)])
        OLS_R2.append([R2(z_train, z_tilde), R2(z_test, z_pred)])

        opacity = 0.4
        if __name__ == "__main__":  # if run bg_taskabc.py
            opacity = 0.2
            surf, ax, axR, axL = plot(i, deg, lambdas, fig,fig_Ridge,fig_Lasso, x,y,z, x_train,y_train,x_test,y_test, z_pred,z_tilde, z_pred_Ridge,z_tilde_Ridge, z_pred_Lasso,z_tilde_Lasso, alpha = opacity)
            if deg[i] == 4: # Plotting Appendix-plot Lasso Feature Selection
                surf, axL2 = plot_Lasso_Feature_selection(i, deg, lambdas, fig_Lasso_Appendix, x,y,z,x_train,y_train,x_test,y_test, z_pred_Lasso,z_tilde_Lasso)
            
            colorbarL = fig_Lasso_Appendix.colorbar(surf, ax=axL2, shrink=0.5, aspect=5, pad = 0.01)
            colorbarL.ax.set_position([0.8, 0.2, 0.7, 0.2])

        # Plotting for different polynomial degree # correspond to lowest MSE
        elif deg[i] == 21: 
            surf, ax  = plot_OLS(i, deg,fig, x,y,z, x_train,y_train,x_test,y_test, z_pred,z_tilde, alpha = opacity)  
            surf, axR = plot_Ridge(i, deg, lambdas, fig_Ridge, x,y,z,x_train,y_train,x_test,y_test, z_pred_Ridge,z_tilde_Ridge, alpha = opacity)  
        elif deg[i] == 26:
            surf, axL = plot_Lasso(i, deg, lambdas, fig_Lasso, x,y,z,x_train,y_train,x_test,y_test, z_pred_Lasso,z_tilde_Lasso,alpha = opacity)        


        


    # Add a color bar which maps values to colors.
    colorbar = fig.colorbar(surf, ax = ax, shrink=0.5, aspect=5, pad = 0.0001)
    colorbar.ax.set_position([0.8, 0.2, 0.7, 0.2])

    colorbarR = fig_Ridge.colorbar(surf, ax=axR, shrink=0.5, aspect=5, pad = 0.001)
    colorbarR.ax.set_position([0.8, 0.2, 0.7, 0.2])

    colorbarL = fig_Lasso.colorbar(surf, ax=axL, shrink=0.5, aspect=5, pad = 0.01)
    colorbarL.ax.set_position([0.8, 0.2, 0.7, 0.2])


    if __name__ == "__main__": # if run bg_taskabc.py
        fig.savefig("OLS.png")
        fig_Ridge.savefig("Ridge.png")
        fig_Lasso.savefig("Lasso.png")
        fig_Lasso_Appendix.savefig("Lasso_Appendix.png")
    else: # if run task_g.py
        fig.savefig("OLS_Real.png")
        fig_Ridge.savefig("Ridge_Real.png")
        fig_Lasso.savefig("Lasso_Real.png")



    if bool_info == True:
        info(deg, OLSbeta_list, OLS_MSE, OLS_R2, lambdas, Ridgebeta_list, Ridge_MSE, Ridge_R2, Lassobeta_list, Lasso_MSE, Lasso_R2)

    







       

    

def plot(i, deg, lambdas, fig,fig_Ridge,fig_Lasso, x,y,z, x_train,y_train,x_test,y_test, z_pred,z_tilde, z_pred_Ridge,z_tilde_Ridge, z_pred_Lasso,z_tilde_Lasso, alpha = .5):
    # ------------------------------------- Plotting -------------------------------------
    # OLS
    ax = fig.add_subplot(2,3,i+1, projection='3d')

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, alpha = 0.7)

    pred = ax.scatter(x_test, y_test, z_pred, color='r', s=10, alpha=0.5, label='z_Pred')
    tilde = ax.scatter(x_train, y_train, z_tilde, color='g', s=10, alpha=0.5, label='z_Tilde')

    fig.suptitle('OLS Regression', fontsize=16)
    ax.set_title(f'Polynomial Degree {deg[i]}', fontsize=10)

    # For organizing, too much info in plots
    if i == 0:
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.legend(loc='upper left', fontsize='small')


    # Ridge
    axR = fig_Ridge.add_subplot(2,3,i+1, projection='3d')

    surf = axR.plot_surface(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, alpha = 0.7)
    # Plot only for highest lambda.
    # Chaos to spot differences by plot.
    # Will look at MSE and R2
    pred = axR.scatter(x_test, y_test, z_pred_Ridge[-1], color='r', s=10, alpha=0.5, label=f'z_Pred, lmd = {lambdas[-1]}') 
    tilde = axR.scatter(x_train, y_train, z_tilde_Ridge[-1], color='g', s=10, alpha=0.5, label=f'z_Tilde, lmd = {lambdas[-1]}')

    fig_Ridge.suptitle(f'Ridge Regression\n  Lambda = {lambdas[-1]}', fontsize=16)
    axR.set_title(f'Polynomial Degree {deg[i]}', fontsize=10)
    if i == 0:
        axR.set_xlabel('X axis')
        axR.set_ylabel('Y axis')
        axR.set_zlabel('Z axis')
        axR.legend(loc='upper left', fontsize='small')
    

    # Lasso
    axL = fig_Lasso.add_subplot(2,3,i+1, projection='3d')

    surf = axL.plot_surface(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, alpha = 0.7)
    # Plot only for one lambda.
    # Chaos to spot differences by plot.
    # Look at MSE and R2

    k = -3 # Choose which lambda to plot
    pred = axL.scatter(x_test, y_test, z_pred_Lasso[k], color='r', s=10, alpha=0.5, label=f'z_Pred, lmd = {lambdas[k]}') 
    tilde = axL.scatter(x_train, y_train, z_tilde_Lasso[k], color='g', s=10, alpha=0.5, label=f'z_Tilde, lmd = {lambdas[k]}')
    fig_Lasso.suptitle(f'Lasso Regression\n  Lambda = {lambdas[k]}', fontsize=16)

    axL.set_title(f'Polynomial Degree {deg[i]}', fontsize=10)
    if i == 0:
        axL.set_xlabel('X axis')
        axL.set_ylabel('Y axis')
        axL.set_zlabel('Z axis')
        axL.legend(loc='upper left', fontsize='small')

    return surf, ax, axR, axL

def plot_OLS(i, deg, fig, x,y,z,x_train,y_train,x_test,y_test, z_pred,z_tilde, alpha):

    ax = fig.add_subplot(1,1,1, projection='3d')

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, alpha = 0.7)

    pred = ax.scatter(x_test, y_test, z_pred, color='r', s=0.1, alpha=alpha, label='z_Pred')
    tilde = ax.scatter(x_train, y_train, z_tilde, color='g', s=0.1, alpha=alpha, label='z_Tilde')

    fig.suptitle('OLS Regression, Lausanne', fontsize=16)
    ax.set_title(f'Polynomial Degree {deg[i]}', fontsize=10)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend(loc='upper left', fontsize='small')

    return surf, ax

def plot_Ridge(i, deg, lambdas, fig_Ridge, x,y,z,x_train,y_train,x_test,y_test, z_pred_Ridge,z_tilde_Ridge, alpha):
    
    axR = fig_Ridge.add_subplot(1,1,1, projection='3d')

    surf = axR.plot_surface(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, alpha = 0.7)

    # Plot only for best (by eye) lambda.
    k = 1
    pred = axR.scatter(x_test, y_test, z_pred_Ridge[k], color='r', s=0.1, alpha=alpha, label=f'z_Pred, lmd = {lambdas[k]}') 
    tilde = axR.scatter(x_train, y_train, z_tilde_Ridge[k], color='g', s=0.1, alpha=alpha, label=f'z_Tilde, lmd = {lambdas[k]}')

    fig_Ridge.suptitle(f'Ridge Regression, Lausanne\n  Lambda = {lambdas[k]}', fontsize=16)
    axR.set_title(f'Polynomial Degree {deg[i]}', fontsize=10)

    axR.set_xlabel('X axis')
    axR.set_ylabel('Y axis')
    axR.set_zlabel('Z axis')
    axR.legend(loc='upper left', fontsize='small')
    return surf, axR

def plot_Lasso(i, deg, lambdas, fig_Lasso, x,y,z,x_train,y_train,x_test,y_test, z_pred_Lasso,z_tilde_Lasso, alpha):
    
    axL = fig_Lasso.add_subplot(1,1,1, projection='3d')

    surf = axL.plot_surface(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, alpha = 0.7)

    # Plot only for best (by eye) lambda.
    k = 0 # Choose which lambda to plot
    pred = axL.scatter(x_test, y_test, z_pred_Lasso[k], color='r', s=0.1, alpha=alpha, label=f'z_Pred, lmd = {lambdas[k]}') 
    tilde = axL.scatter(x_train, y_train, z_tilde_Lasso[k], color='g', s=0.1, alpha=alpha, label=f'z_Tilde, lmd = {lambdas[k]}')
    fig_Lasso.suptitle(f'Lasso Regression, Lausanne\n  Lambda = {lambdas[k]}', fontsize=16)

    axL.set_title(f'Polynomial Degree {deg[i]}', fontsize=10)
    axL.set_xlabel('X axis')
    axL.set_ylabel('Y axis')
    axL.set_zlabel('Z axis')
    axL.legend(loc='upper left', fontsize='small')

    return surf, axL

# Illustrating how L1-penalty works
def plot_Lasso_Feature_selection(i, deg, lambdas, fig_Lasso_Appendix, x,y,z,x_train,y_train,x_test,y_test, z_pred_Lasso,z_tilde_Lasso):
    for j in range(len(lambdas)):
        axL = fig_Lasso_Appendix.add_subplot(2,3,j+1, projection='3d')

        surf = axL.plot_surface(x, y, z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False, alpha = 0.7)

        k = j # Choose which lambda to plot
        pred = axL.scatter(x_test, y_test, z_pred_Lasso[k], color='r', s=10, alpha=0.5, label=f'z_Pred') 
        tilde = axL.scatter(x_train, y_train, z_tilde_Lasso[k], color='g', s=10, alpha=0.5, label=f'z_Tilde')
        fig_Lasso_Appendix.suptitle(f'Lasso Regression\n  Polynomial Degree {deg[i]}', fontsize=16)

        axL.set_title(f'Lambda = {lambdas[k]}', fontsize=10)
        if j == 0:
            axL.set_xlabel('X axis')
            axL.set_ylabel('Y axis')
            axL.set_zlabel('Z axis')
            axL.legend(loc='upper left', fontsize='small')
        
    return surf, axL


    
def info(deg, OLSbeta_list, OLS_MSE, OLS_R2, lambdas, Ridgebeta_list, Ridge_MSE, Ridge_R2, Lassobeta_list, Lasso_MSE, Lasso_R2):
    # ------------------------------------------------ PRINTING INFO ------------------------------------------------------
    want_beta = False #True # If want to see beta-values

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
                print(f'Beta = {Ridgebeta_list[i*len(lambdas) +j].T} \n')

    
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




    #------------------------------------------------- Plotting Errors for model complexity  -----------------------------------------------------
    reg_list = ["OLS", "Ridge", "Lasso"]


    fig_MSE = plt.figure(figsize=(10,7))
    fig_MSE.suptitle('MSE vs Polynomal Complexity', fontsize=16)

    fig_R2 = plt.figure(figsize=(10,7))
    fig_R2.suptitle('R2 vs Polynomal Complexity', fontsize=16)

    for i in range(len(reg_list)): 
        ax = fig_MSE.add_subplot(1,3,i+1)
        ax2 = fig_R2.add_subplot(1,3,i+1)

        if reg_list[i] == "OLS":
            ax.plot(deg, OLS_MSE, label=['Train', 'Test'])
            ax2.plot(deg, OLS_R2, label=['Train', 'Test'])
        
        elif reg_list[i] == "Ridge":
            for j in range(len(lambdas)):
                ax.plot(deg, Ridge_MSE[j::5], label=[fr'Train lmd: $10^{{{np.log10(lambdas[j]):.0f}}}$', fr'Test lmd: $10^{{{np.log10(lambdas[j]):.0f}}}$']) 
                ax2.plot(deg, Ridge_R2[j::5], label=[fr'Train lmd: $10^{{{np.log10(lambdas[j]):.0f}}}$', fr'Test lmd: $10^{{{np.log10(lambdas[j]):.0f}}}$']) 
                # lokking from correct lambda-value

        else:
            for j in range(len(lambdas)):
                ax.plot(deg, Lasso_MSE[j::5], label=[fr'Train lmd: $10^{{{np.log10(lambdas[j]):.0f}}}$', fr'Test lmd: $10^{{{np.log10(lambdas[j]):.0f}}}$'])
                ax2.plot(deg, Lasso_R2[j::5], label=[fr'Train lmd: $10^{{{np.log10(lambdas[j]):.0f}}}$', fr'Test lmd: $10^{{{np.log10(lambdas[j]):.0f}}}$']) 
                # lokking from correct lambda-value



        ax.set_title(f'{reg_list[i]}', fontsize=10)
        ax.set_xlabel('Polynomial Degree')
        if i == 0:
            ax.set_ylabel('MSE')
        ax.set_ylim(0,np.max(Lasso_MSE))
        ax.grid()
        ax.legend(loc='upper left', fontsize='small')
        

        ax2.set_title(f'{reg_list[i]}', fontsize=10)
        ax2.set_xlabel('Polynomial Degree')
        if i == 0:
            ax2.set_ylabel('R2')
        ax2.set_ylim(-.1,1)
        ax2.grid()
        ax2.legend(loc='lower left', fontsize='small')

    fig_MSE.tight_layout()
    if __name__ == "__main__": # if run bg_taskabc.py
        fig_MSE.savefig("MSE_task_abc.png")
        fig_R2.savefig("R2_task_abc.png")
    else: # if run task_g.py
        fig_MSE.suptitle('MSE vs Polynomal Complexity, Lausanne', fontsize=16)
        fig_MSE.savefig("MSE_Real.png")
    
    plt.show()





if __name__ == "__main__":
    #----------------------------------- Making data -------------------------------------------------------------------------------
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    x, y = np.meshgrid(x,y)

    z = FrankeFunction(x, y)
    deg_max = 6  # up to degree 5
    
    
    regression(x,y,z, deg_max, bool_info = True)
    #deg, OLSbeta_list, OLS_MSE, OLS_R2, lambdas, Ridgebeta_list, Ridge_MSE, Ridge_R2, Lassobeta_list, Lasso_MSE, Lasso_R2



















