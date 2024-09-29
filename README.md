# FYS-STK_Project1

Codes:
  bg_taskabc.py  -   OLS, Ridge, Lasso regression on Franke's Function   
  task_g.py      -   OLS, Ridge, Lasso regression on real topografic data from Lausanne (datasset: Lausanne.tif)

Functions:
  Regression-Function (found in bg_taskabc.py)
    - input variables: x-, y-, z-positions of surface
    - input variables: maximum polynomial degree of model
    - input variables: bool if want printed out info (.log-file): lambda, MSE, R2, beta (reg. coeff.)
      note: want beta-values must set "want_beta = True" (line 437)
    - structure: found in Project1.tex
    
  Ole Petter: putt in some important functions structure


How to run: 
  $ nohup python3 bg_taskabc.py > output_test.log 2>&1 &
  $ nohup python3 task_g.py > output_real.log 2>&1 &


