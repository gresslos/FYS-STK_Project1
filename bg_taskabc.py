import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)




np.random.seed(1)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


z = FrankeFunction(x, y)

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('FrankeFunction.png')
plt.show()






n = 100
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

# Defining Design matrix (The Vermonde-Matrix) 
deg = 5 #degree of polynomial


X = np.zeros((len(x), deg))
# PS: not include intercept in design matrix, will scale (centered values)
for d in range(deg):
    X[:,d] = x.flatten() ** (d+1)


# Split data in training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
# give: test 1/5 of data,  train = 4/5 af data

# Model Training
X_train_mean = np.mean(X_train)
y_train_mean = np.mean(y_train)
X_train_scaled = X_train - X_train_mean
y_train_scaled = y_train - y_train_mean
# Model predictions
X_test_scaled = X_test - X_train_mean



# ------------------------- MANUALLY OLS ---------------------------------------------
OLSbeta = np.linalg.inv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ y_train_scaled
y_train_OLS = X_train_scaled @ OLSbeta + y_train_mean # adding intercepts


y_test_OLS = X_test_scaled @ OLSbeta + y_train_mean # adding intercepts


# Sort the training and test data and corresponding predictions
sorted_indices = np.argsort(X_train_scaled[:,0].flatten())
X_train_sort = X_train_scaled[sorted_indices]
y_train_OLS_sort = y_train_OLS[sorted_indices]
y_train_sort = y_train_scaled[sorted_indices]

sorted_indices = np.argsort(X_test_scaled[:,0].flatten())
X_test_sort = X_test_scaled[sorted_indices]
y_test_OLS_sort = y_test_OLS[sorted_indices]
y_test_sort = y_test[sorted_indices]


# Finding values for plotting (Fist column in Design-Matrix X)
# Adding mean-value for plotting
x_train = X_train_sort[:,0] + X_train_mean 
x_test = X_test_sort[:,0] + X_train_mean


fig, [ax1, ax2] = plt.subplots(1, 2,figsize = (15,8))
fig.suptitle(f'Polynomial Degree {deg}', fontsize=16)
ax1.set_title(f'OLS Regression (Manually)')
ax1.plot(x,y, 'bo', ms = 5, label='data')
ax1.plot(x_train, y_train_OLS_sort, 'r', label='Regression Training')
ax1.plot(x_test, y_test_OLS_sort, 'g', label='Regression Test')
ax1.legend()
plt.plot()
plt.savefig(f'OLS.png')


"""
# ----------------------------- USING SICKIT-LEARN------------------------------
# not included intercept in design matrix X
linreg = LinearRegression(fit_intercept=False).fit(X_train, y_train)


y_train_OLS = linreg.predict(X_train_sort) + y_train_mean
y_test_OLS = linreg.predict(X_test_sort) + y_train_mean


#ax2 = fig.add_subplot(2,1,2)
ax2.set_title('OLS Regression (Scikit-Learn)')
ax2.plot(x,y, 'bo', ms = 5, label='data')
ax2.plot(x_train, y_train_OLS, 'r', label='Regression Training')
ax2.plot(x_test, y_test_OLS, 'g', label='Regression Test')
ax2.legend()
plt.plot()
plt.savefig(f'uke_36_fig2{deg}.png')
print()

"""

# ----------------------------- MSE ---------------------
MSETrain_OLS = MSE(y_train_sort + y_train_mean, y_train_OLS_sort)
MSETest_OLS = MSE(y_test_sort, y_test_OLS_sort)

R2Train_OLS = R2(y_train_sort + y_train_mean, y_train_OLS_sort)
R2Test_OLS = R2(y_test_sort, y_test_OLS_sort)

print(f'Polynomial Degree {deg}')
print(f"Training MSE for OLS = {MSETrain_OLS:.2f} \nTest MSE OLS = {MSETest_OLS:.2f} \n")
print(f"Training R2 for OLS = {R2Train_OLS:.2f} \nTest R2 OLS = {R2Test_OLS:.2f}")


    
    