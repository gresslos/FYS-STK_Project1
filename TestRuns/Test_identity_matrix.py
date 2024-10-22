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



# Test_1D was startingpoint of code-implementation in 1D-space
def test_1D(test_X_is_identity_matrix = True):

    # ----------------------------------------- Testing y_tilde = y_train for X = I --------------------------------------------------------
    # -----------------------------------------   Will expect y = y_tilde for OLS   --------------------------------------------------------
    # -----------------------------------------        MSE = 0 and R2 = 1           --------------------------------------------------------
    #test_X_is_identity_matrix  : Turn on / off to test if y_train = y_tilde when design matrix = Identity Matrix


    # Make data.
    x = np.arange(0, 1, 0.05)
    y = 1 # Testing one dimentional
    
    z = FrankeFunction(x, y)

    # Split data in training and test data
    x_train, x_test, z_train, z_test = train_test_split(x, z, test_size=0.25) 
    # give: test 1/4 of data,  train = 3/4 af data

    deg = np.arange(1,6) #degree of polynomial
    fig = plt.figure(figsize=(8,8)) # OLS
    fig_Ridge = plt.figure(figsize=(8,8)) # Ridge
    fig_Lasso = plt.figure(figsize=(8,8)) # Lasso

    for i in range(len(deg)):   
        # ------------------------------- Making Design Matrix -----------------------
        X_train = np.zeros((len(x_train), deg[i]))
        X_test = np.zeros((len(x_test), deg[i]))
        # PS: not include intercept in design matrix, will scale (centered values)
        col = 0
        for dx in range(1,deg[i] + 1):
            X_train[:,col] = (x_train ** dx)
            X_test[:,col] = (x_test ** dx)
            col += 1

        if test_X_is_identity_matrix:
            X_train = np.eye(X_train.shape[0],X_train.shape[0])
            X_test = np.eye(X_train.shape[0],X_train.shape[0])

        

        
        #----------------------------------------   Scaling -----------------------------
        if test_X_is_identity_matrix == False:
            stdsc = StandardScaler() # For x- and y-vals
            X_train = stdsc.fit_transform(X_train)
            X_test = stdsc.transform(X_test)

            stdsc_z = StandardScaler() # For z-vals
            z_train = stdsc_z.fit_transform(z_train.reshape(-1,1)) 
            z_test = stdsc_z.transform(z_test.reshape(-1,1))
            # reshape to get 2D array
            # .transform() expect 2D array
        


        # ------------------------- MANUALLY OLS ---------------------------------------------
        


        OLSbeta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train
        # and then make the prediction
        z_tilde = X_train @ OLSbeta 
        z_pred = X_test @ OLSbeta 



        # ------------------------- MANUALLY RIDGE + SCIKIT-LEARN LASSO ----------------------
        if test_X_is_identity_matrix:
            I = np.eye(X_train.shape[0],X_train.shape[0])
        else:
            I = np.eye(deg[i],deg[i])

        lambdas = [0.0001,0.001,0.01,0.1,1]

        if test_X_is_identity_matrix:
            z_tilde_Ridge = np.zeros((len(lambdas), X_train.shape[0])) # changing dimentions to fit
            z_pred_Ridge = np.zeros((len(lambdas), X_test.shape[0]))
        else:
            z_tilde_Ridge = np.zeros((len(lambdas), X_train.shape[0], 1))
            z_pred_Ridge = np.zeros((len(lambdas), X_test.shape[0], 1))


        z_tilde_Lasso = np.zeros((len(lambdas), X_train.shape[0]))
        z_pred_Lasso = np.zeros((len(lambdas), X_test.shape[0]))

        for j in range(len(lambdas)):
            lmb = lambdas[j]
            Ridgebeta = np.linalg.inv(X_train.T @ X_train + lmb*I) @ X_train.T @ z_train
            LassoReg = Lasso(lmb,fit_intercept=False) # Not include intercept
            LassoReg.fit(X_train, z_train)
            # and then make the prediction
            z_tilde_Ridge[j] = X_train @ Ridgebeta
            z_pred_Ridge[j] = X_test @ Ridgebeta

            z_tilde_Lasso[j] = LassoReg.predict(X_train)
            z_pred_Lasso[j] = LassoReg.predict(X_test)





        if test_X_is_identity_matrix == False:
            # ------------------------ RESCALING ------------------------------------------------
            # Rescale with StandardScaler
            
            # General variables
            X_test = stdsc.inverse_transform(X_test)      
            X_train = stdsc.inverse_transform(X_train)

            z_test = stdsc_z.inverse_transform(z_test)
            z_train = stdsc_z.inverse_transform(z_train)

            # OLS
            z_pred = stdsc_z.inverse_transform(z_pred)
            z_tilde = stdsc_z.inverse_transform(z_tilde) 

            # Ridge
            z_pred_Ridge = [stdsc_z.inverse_transform(z_pred_Ridge[j]) for j in range(len(lambdas))]
            z_tilde_Ridge = [stdsc_z.inverse_transform(z_tilde_Ridge[j]) for j in range(len(lambdas))]

            # Lasso
            z_pred_Lasso = stdsc_z.inverse_transform(z_pred_Lasso)
            z_tilde_Lasso = stdsc_z.inverse_transform(z_tilde_Lasso)

        

        # ----------------------------- Test MSE for OLS ---------------------
        if test_X_is_identity_matrix:
            MSETrain_OLS = MSE(z_train, z_tilde)
            R2Train_OLS = R2(z_train, z_tilde)

            print(f'       1D:            OLS Regressions \n    Polynomial Degree {deg[i]}')
            print(f"Training MSE = {MSETrain_OLS:.2e} \n")
            print(f"Training R2 = {R2Train_OLS:.2e}\n")
        
        else:
            MSETrain_OLS = MSE(z_train, z_tilde)
            MSETest_OLS = MSE(z_test, z_pred)

            R2Train_OLS = R2(z_train, z_tilde)
            R2Test_OLS = R2(z_test, z_pred)

            print(f'Polynomial Degree {deg[i]}')
            print(f"Training MSE for OLS = {MSETrain_OLS:.2e} \nTest MSE OLS = {MSETest_OLS:.2e} \n")
            print(f"Training R2 for OLS = {R2Train_OLS:.2e} \nTest R2 OLS = {R2Test_OLS:.2e}\n")
        


        # Sort the training and test data and corresponding predictions
        sorted_indices = np.argsort(X_train[:,0].flatten())
        X_train_sort = X_train[sorted_indices]
        z_train_sort = z_train[sorted_indices]
        z_tilde_sort = z_tilde[sorted_indices]
        z_tilde_Ridge_sort = [z_tilde_Ridge[j][sorted_indices] for j in range(len(lambdas))]
        z_tilde_Lasso_sort = [z_tilde_Lasso[j][sorted_indices] for j in range(len(lambdas))]
        
        if test_X_is_identity_matrix == False:
            sorted_indices = np.argsort(X_test[:,0].flatten())
            X_test_sort = X_test[sorted_indices]
            z_test_sort = z_test[sorted_indices]
            z_pred_sort = z_pred[sorted_indices]
            z_pred_Ridge_sort = [z_pred_Ridge[j][sorted_indices] for j in range(len(lambdas))]
            z_pred_Lasso_sort = [z_pred_Lasso[j][sorted_indices] for j in range(len(lambdas))]

            x_test_sort = X_test_sort[:,0]
        

        # Finding values for plotting (Fist column in Design-Matrix X)
        x_train_sort = X_train_sort[:,0] 
        


        


        #--------------------------------- Ploting -----------------------------------------------
        # OLS
        ax = fig.add_subplot(2,3,i+1)
        fig.suptitle('OLS Regression')
        ax.set_title(f'Pol.degree = {deg[i]}')

        surf = ax.plot(x, z, label='FrankeFunction') 
        if test_X_is_identity_matrix:
            reg = ax.scatter(x_train, z_tilde, label=f'Tilde: z(x)')
        else:
            reg = ax.plot(x_test_sort, z_pred_sort, label=f'Predicted z(x)')
        ax.set_ylim(.0, .3)
        ax.grid()
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Ridge
        axR = fig_Ridge.add_subplot(2,3,i+1)
        fig_Ridge.suptitle('Ridge Regression')
        axR.set_title(f'Pol.degree = {deg[i]}')

        surf = axR.plot(x, z, label='FrankeFunction') 
        if test_X_is_identity_matrix:
            reg = [axR.scatter(x_train, z_tilde_Ridge[j], label=f'z_pred(x), lmd = {lambdas[j]}') for j in range(len(lambdas))]
        else:
            reg = [axR.plot(x_test_sort, z_pred_Ridge_sort[j], label=f'z_pred(x), lmd = {lambdas[j]}') for j in range(len(lambdas))]
        axR.set_ylim(.0, .3)
        axR.grid()
        axR.set_xlabel('x')
        axR.set_ylabel('y')

        # Lasso
        axL = fig_Lasso.add_subplot(2,3,i+1)
        fig_Lasso.suptitle('Ridge Regression')
        axL.set_title(f'Pol.degree = {deg[i]}')

        surf = axL.plot(x, z, label='FrankeFunction') 
        if test_X_is_identity_matrix:
            reg = [axL.scatter(x_train, z_tilde_Lasso[j], label=f'z_pred(x), lmd = {lambdas[j]}') for j in range(len(lambdas))]
        else:
            reg = [axL.plot(x_test_sort, z_pred_Lasso_sort[j], label=f'z_pred(x), lmd = {lambdas[j]}') for j in range(len(lambdas))]
        axL.set_ylim(.0, .3)
        axL.grid()
        axL.set_xlabel('x')
        axL.set_ylabel('y')

    axR.legend()
    axL.legend()
    fig_Ridge.tight_layout()
    fig_Lasso.tight_layout()

    fig.savefig('FrankeFunction_1D_regression.png')
    plt.show()




# 2D model
def regression(x, y, z, deg_max = 6, bool_info = False, test_X_is_identity_matrix = True):

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
        fig_best  = plt.figure(figsize=(12,7))
        fig_Lasso_Appendix = plt.figure(figsize=(12,7))

        
    else:                        # if run task_g.py
        fig       = plt.figure(figsize=(7,7))
        fig_Ridge = plt.figure(figsize=(7,7))
        fig_Lasso = plt.figure(figsize=(7,7))
    
    # -------------------------------------- Scaling + Regression -----------------------------------
    for i in range(len(deg)):
        # Making Design Matrix Phi 
        Phi_train = Design_Matrix_2D(deg[i], X_train)
        Phi_test  = Design_Matrix_2D(deg[i], X_test)

        #----------------------------------------   Scaling -----------------------------------------
        stdsc = StandardScaler() # For x- and y-vals
        Phi_train = stdsc.fit_transform(Phi_train)
        Phi_test = stdsc.transform(Phi_test)

        stdsc_z = StandardScaler() # For z-vals
        z_train = stdsc_z.fit_transform(z_train.reshape(-1,1))
        z_test = stdsc_z.transform(z_test.reshape(-1,1))
        # reshape to get 2D array
        # function .fit_transform() and .transform() expect 2D array 


        # ------------------------------ Design Matrix = Identity Matrix ----------------------------
        if test_X_is_identity_matrix:
            Phi_train  = np.eye(Phi_train.shape[0],Phi_train.shape[0])
            Phi_test   = Phi_train # np.eye(Phi_train.shape[0],Phi_train.shape[0])

        #-----------------------------------------   OLS - Regression    -----------------------------
        OLSbeta = np.linalg.pinv(Phi_train.T @ Phi_train) @ Phi_train.T @ z_train
        z_pred = Phi_test @ OLSbeta
        z_tilde = Phi_train @ OLSbeta

        # Adding OLSbeta to list 
        OLSbeta_list.append(OLSbeta)


        # ---------------------------------- MANUALLY RIDGE + SCIKIT-LEARN LASSO ----------------------
        num_terms = int((deg[i] + 1) * (deg[i] + 2) / 2 - 1) # Numbers from Design_Matrix_2D()
        I = np.eye(num_terms,num_terms)

        if test_X_is_identity_matrix:
            I = np.eye(Phi_train.shape[0],Phi_train.shape[0])


        lambdas = [0.0001,0.001,0.01,0.1,1]

        z_tilde_Ridge = np.zeros((len(lambdas), X_train.shape[0],1))
        z_pred_Ridge = np.zeros((len(lambdas), X_test.shape[0], 1))

        z_tilde_Lasso = np.zeros((len(lambdas), X_train.shape[0], 1))
        z_pred_Lasso = np.zeros((len(lambdas), X_test.shape[0], 1))

        if test_X_is_identity_matrix:
            z_tilde_Ridge = np.zeros((len(lambdas), Phi_train.shape[0],1))
            z_pred_Ridge = np.zeros((len(lambdas), Phi_train.shape[0], 1))

            z_tilde_Lasso = np.zeros((len(lambdas), Phi_train.shape[0], 1))
            z_pred_Lasso = np.zeros((len(lambdas), Phi_train.shape[0], 1))

        for j in range(len(lambdas)):
            lmb = lambdas[j]

            Ridgebeta = np.linalg.pinv(Phi_train.T @ Phi_train + lmb*I) @ Phi_train.T @ z_train

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
        if test_X_is_identity_matrix:
            x_train = X_train[:,0].reshape(-1,1)
            y_train = X_train[:,1].reshape(-1,1)

            x_test = X_test[:,0]
            y_test = X_test[:,1]

        

        else:
            #Reverse wScaling with StandardScaler.inverse_transform()
            Phi_test = stdsc.inverse_transform(Phi_test)
            x_test = Phi_test[:,0].reshape(-1,1)
            y_test = Phi_test[:,1].reshape(-1,1)

            Phi_train = stdsc.inverse_transform(Phi_train)
            x_train = Phi_train[:,0].reshape(-1,1)
            y_train = Phi_train[:,1].reshape(-1,1)


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
        if test_X_is_identity_matrix:
            print(f'      2D:          OLS Regression \n   Polynomial Degree {deg[i]}\n')
            print(f"Training MSE for OLS = {MSE(z_train, z_tilde):10.2e} \n")
            print(f"Training R2 for OLS = {R2(z_train, z_tilde):10.2e} \n\n")

        else:
            for j in range(len(lambdas)):
                # Ridge
                Ridge_MSE.append([MSE(z_train, z_tilde_Ridge[j]),MSE(z_test, z_pred_Ridge[j])])
                Ridge_R2.append([R2(z_train, z_tilde_Ridge[j]), R2(z_test, z_pred_Ridge[j])])

                # Lasso
                Lasso_MSE.append([MSE(z_train, z_tilde_Lasso[j]),MSE(z_test, z_pred_Lasso[j])])
                Lasso_R2.append([R2(z_train, z_tilde_Lasso[j]), R2(z_test, z_pred_Lasso[j])])
        

            OLS_MSE.append([MSE(z_train, z_tilde), MSE(z_test, z_pred)])
            OLS_R2.append([R2(z_train, z_tilde), R2(z_test, z_pred)])
        
        #-------------------------- Plot-funtions ----------------------------------------
        opacity = 0.6
        surf, ax, axR, axL = 0,0,0,0
        if __name__ == "__main__":  # if run bg_taskabc.py
            opacity = 0.2
            surf, ax, axR, axL,axB = plot(i, deg, lambdas, fig,fig_Ridge,fig_Lasso, fig_best, x,y,z, x_train,y_train,x_test,y_test, z_pred,z_tilde, z_pred_Ridge,z_tilde_Ridge, z_pred_Lasso,z_tilde_Lasso,test_X_is_identity_matrix, alpha = opacity)
            
    fig.savefig("OLS.png")
    #plt.show()



    if bool_info == True:
        info(deg, OLSbeta_list, OLS_MSE, OLS_R2, lambdas, Ridgebeta_list, Ridge_MSE, Ridge_R2, Lassobeta_list, Lasso_MSE, Lasso_R2)


def plot(i, deg, lambdas, fig,fig_Ridge,fig_Lasso, fig_best, x,y,z, x_train,y_train,x_test,y_test, z_pred,z_tilde, z_pred_Ridge,z_tilde_Ridge, z_pred_Lasso,z_tilde_Lasso,test_X_is_identity_matrix, alpha = .5):
    # ------------------------------------- Plotting -------------------------------------
    # OLS
    ax = fig.add_subplot(2,3,i+1, projection='3d')

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False, alpha = 0.7)
    
    if test_X_is_identity_matrix == False:
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
    if test_X_is_identity_matrix == False:
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
    if test_X_is_identity_matrix == False:
        pred = axL.scatter(x_test, y_test, z_pred_Lasso[k], color='r', s=10, alpha=0.5, label=f'z_Pred, lmd = {lambdas[k]}') 
    tilde = axL.scatter(x_train, y_train, z_tilde_Lasso[k], color='g', s=10, alpha=0.5, label=f'z_Tilde, lmd = {lambdas[k]}')
    fig_Lasso.suptitle(f'Lasso Regression\n  Lambda = {lambdas[k]}', fontsize=16)

    axL.set_title(f'Polynomial Degree {deg[i]}', fontsize=10)
    if i == 0:
        axL.set_xlabel('X axis')
        axL.set_ylabel('Y axis')
        axL.set_zlabel('Z axis')
        axL.legend(loc='upper left', fontsize='small')


    # -------------------------------- Plotting best fit ------------------------------
    ax1 = 0
    # OLS
    if deg[i] == 5:
        ax1 = fig_best.add_subplot(1,3,3, projection='3d')

        surf = ax1.plot_surface(x, y, z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False, alpha = 0.7)

        pred = ax1.scatter(x_test, y_test, z_pred, color='r', s=10, alpha=0.5, label='z_Pred')
        tilde = ax1.scatter(x_train, y_train, z_tilde, color='g', s=10, alpha=0.5, label='z_Tilde')

        fig_best.suptitle('Best Regressions', fontsize=16)
        ax1.set_title(f'Polynomial Degree {deg[i]} \n OLS Regression', fontsize=10)
        ax1.set_zlim(0,1.25)

    # Ridge
    elif deg[i] == 4:
        k = 1 # index for best lambda = 0.001
        ax2 = fig_best.add_subplot(1,3,2, projection='3d')

        surf = ax2.plot_surface(x, y, z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False, alpha = 0.7)

        pred = ax2.scatter(x_test, y_test, z_pred_Ridge[k], color='r', s=10, alpha=0.5, label='z_Pred')
        tilde = ax2.scatter(x_train, y_train, z_tilde_Ridge[k], color='g', s=10, alpha=0.5, label='z_Tilde')

        ax2.set_title(f'Polynomial Degree {deg[i]} \n Ridge Regression \n  Lambda = {lambdas[k]}', fontsize=10)
        ax2.set_zlim(0,1.25)

    # Lasso
    elif deg[i] == 3:
        k = 1 # index for best lambda = 0.001
        ax3 = fig_best.add_subplot(1,3,1, projection='3d')

        surf = ax3.plot_surface(x, y, z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False, alpha = 0.7)

        pred = ax3.scatter(x_test, y_test, z_pred_Lasso[k], color='r', s=10, alpha=0.5, label='z_Pred')
        tilde = ax3.scatter(x_train, y_train, z_tilde_Lasso[k], color='g', s=10, alpha=0.5, label='z_Tilde')

        ax3.set_title(f'Polynomial Degree {deg[i]} \n Lasso Regression \n  Lambda = {lambdas[k]}', fontsize=10)

        ax3.set_zlim(0,1.25)
        ax3.set_xlabel('X axis')
        ax3.set_ylabel('Y axis')
        ax3.set_zlabel('Z axis')
        ax3.legend(loc='upper left', fontsize='small')

    return surf, ax, axR, axL, ax1

    
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
        fig_R2.suptitle('R2 vs Polynomal Complexity, Lausanne', fontsize=16)
        fig_R2.savefig("R2_Real.png")
    
    plt.show()



if __name__ == "__main__":
    #----------------------------------- Making data -------------------------------------------------------------------------------
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    x, y = np.meshgrid(x,y)

    z = FrankeFunction(x, y)
    deg_max = 6  # up to degree 5
    
    # Testing the functions
    test_1D(test_X_is_identity_matrix = True)

    regression(x,y,z, deg_max, bool_info = False, test_X_is_identity_matrix = True)
    # NOTES 
        # Change test_X_is_identity_matrix = True / False to see / not see testing with identity matrix
        # bool_info = False, only want OLS traing data
        # Only checking if y_tilde = y_train, therefor no y_pred info
        
