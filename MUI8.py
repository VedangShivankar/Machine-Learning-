import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
dir(digits)
print(digits.data[0])