Test_identity_matrix.py is a copy of bg_taskabc.py, but include:
'test_X_is_identity_matrix = True'- statement. 
This will allow for the code to set the design matrix to the identity matrix.
This will lead to y_tilde = y_train
(Look at y_tilde given split data to test and training data, and compara data (y_train) to fitted model (y_tilde))
The code plots MSE and R2 for y_tilde/y_train which is respectivly 0.00 and 1.00.
