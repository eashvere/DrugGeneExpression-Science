import matplotlib.pyplot as plt
import pickle
with open("validation_lists.pkl", 'rb') as f:
    data = pickle.load(f)

arr = data["val_resp_mae"]
x = range(1, len(arr)+1)
plt.plot(x, arr)
plt.xlabel('Epochs')
plt.ylabel('Value Response MAE')
plt.title('Value Response MAE over Training')
plt.show()

arr = data["val_resp_mse"]
plt.plot(x, arr)
plt.xlabel('Epochs')
plt.ylabel('Value Response MSE')
plt.title('Value Response MSE over Training')
plt.show()

arr = data["val_resp_r2"]
plt.plot(x, arr)
plt.xlabel('Epochs')
plt.ylabel('Value Response R2')
plt.title('Value Response R2 over Training')
plt.show()
