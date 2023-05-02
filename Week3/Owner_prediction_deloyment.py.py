import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import pickle

# Uncomment the following snippet of code to debug problems with finding the .pkl file path
# This snippet of code will exit the program and print the current working directory.
# import os
# exit(os.getcwd())

riding_mover = pickle.load(open("E:/Spring-23/DSP/week3/finalout.pkl", "rb"))

print("\n*****************************************************")
print("* Lawn Owner Prediction *")
print("*****************************************************\n")
Income = float(input("Enter the income of the person: "))
Lot_Size = float(input("Enter the Lot Size of the person: "))

data = pd.DataFrame({'Income': [Income],'Lot_Size':[Lot_Size]})
output = riding_mover.predict(data)
probs = riding_mover.predict_proba(data)
predct = ('Nonowner', 'Owner')
print(f"\nThe Lawn owner prediction model indicates probability of ownership at {probs[0][1]:.4f}, therefore we can predict thst the person is a {predct[output[0]]}.\n")
