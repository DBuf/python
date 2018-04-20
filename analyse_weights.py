import numpy as np
import matplotlib.pyplot as plt

w1=np.load('weights_1.npy')
w2=np.load('weights_3.npy')

print(w1.shape)
#print(w1)
print(w2.shape)
#print(w2)

wc=np.dot(w1,w2)
wc=np.reshape(wc,[60,4])
print(wc.shape)
#print(wc)

# Plot the grid
plt.imshow(np.transpose(np.abs(wc)))
plt.gray()
plt.show()

# Normalize by the input mean
x_train=np.load('x_train.npy')
print(x_train.shape)
print(np.mean(np.mean(x_train,1),0))

plt.imshow(np.transpose(np.abs(wc)*np.mean(np.mean(x_train,1),0)))
plt.gray()
plt.show()

