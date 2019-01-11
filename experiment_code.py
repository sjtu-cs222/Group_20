
# coding: utf-8

# In[ ]:


# select the GPU, you can skip this part
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# In[ ]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# In[ ]:


import numpy as np


# In[ ]:



batch_size = 128
num_classes = 10
epochs = 5

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[ ]:


model = Sequential()
conv1 = Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape)
model.add(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu')
model.add(conv2)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[ ]:


# train part
# can be train several times depends on demand
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          # epochs=3,
          verbose=1,
          validation_data=(x_test, y_test))


# In[ ]:


def print_acc():
    global model
    # evaluation part
    score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# In[ ]:


print_acc()


# In[ ]:


# the selection of the part of the kernel filter varies in stage of the experiments
def hash_3x3(filter:np.array) -> np.array:
    return np.int32(np.array([filter[0,0], filter[0,1], filter[0,2], filter[1,0], filter[2, 0]])
                              * 1000)


# In[ ]:


# experiment 1 starts from here
# be sure to retrain the model or to store and restore the weight matrix for the original accuracy.
conv1w, conv1b = conv1.get_weights()
conv2w, conv2b = conv2.get_weights()
hash_matrix_l1 = np.zeros((conv1w.shape[3], conv1w.shape[2], 5), dtype=np.int32)
hash_matrix_l2 = np.zeros((conv2w.shape[3], conv2w.shape[2], 5), dtype=np.int32)


# In[ ]:


for i in range(32):
    hash_matrix_l1[i][0] = (hash_3x3(conv1w[:,:,0,i]))


# In[ ]:


for i in range(64):
    for j in range(32):
        hash_matrix_l2[i][j] = (hash_3x3(conv2w[:,:,j,i]))


# In[ ]:


# conv2b.shape


# In[ ]:


# replace weights in layer 1 by layer 2
for i1 in range(conv1w.shape[3]):
    # conv1
    next_i = 0
    min_ = 1000
    idx = []
    for i2 in range(64):
        for j2 in range(32):
            tmp = np.sum(np.abs(hash_matrix_l1[i1][0] - hash_matrix_l2[i2][j2]))
            if (tmp < min_):
                min_ = tmp
                idx = [i2, j2]
            if (min_ < 100):
                # print(i1, i2, j2, tmp)
                conv1w[:,:,0,i1] = conv2w[:,:,j2,i2]
                next_i = 1
                break
        if (next_i == 1):
            break
    if (next_i == 0):
        # print(i1, idx, min_)
        if (min_ < 250): # should be determined manually
            conv1w[:,:,0,i1] = conv2w[:,:,idx[1],idx[0]]


# In[ ]:


conv1.set_weights([conv1w, conv1b])


# In[ ]:


print("After weight replacement")
print_acc()


# In[ ]:


# end of experiment 1 #####################################


# In[ ]:


# test part, you can ignore this part
for c in range(31):
    # layer 2; channel c, c+1;
    for i in range(64):
        for j in range(64):
            for k in range(4):
                tmp += np.abs(hash_matrix_l2[i][c][k] - hash_matrix_l2[j][c+1][k])
                if (tmp > 50):
                    break
            if tmp < 50:
                print(c, i, j, tmp)
                next_l = 1
                conv2w[:,:,c,i] = conv2w[:,:,c+1,j]
                break
                # max_ = tmp
    


# In[ ]:


# experiment 2
# train again for second method experiment
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))


# In[ ]:


print_acc()


# In[ ]:


# you can start experiment 2 from here. 
num_basis = 6 # it's tunable
a_s = np.zeros((32, num_basis))
basis = np.random.rand(num_basis,3,3) - 0.5


# In[ ]:


conv1w, conv1b = conv1.get_weights()
conv2w, conv2b = conv2.get_weights()


# In[ ]:


def sum_all(a, b):
        tmp = np.zeros((b.shape[1], b.shape[2]))
        for i in range(b.shape[0]):
            tmp += a[i] * b[i,:,:]
        return tmp


# In[ ]:


def get_efficients(basis:np.array, target:np.array) -> np.array:
    res = (np.random.rand(basis.shape[0]) - 0.5) / 10
    loss = np.sum(np.square(target - sum_all(res, basis)))
    
    # gradient descent method
    dres = np.zeros(basis.shape[0])
    lr = 0.3
    j = 0 # iteration times
    # print(loss)
    while(j < 300):
        tmp_y = sum_all(res, basis)
        for i in range(basis.shape[0]):
            '''
            dres[i] = 0
            for ii in range(3):
                for jj in range(3):
                    dres[i] += 2 * (target[ii][jj] - tmp_y[ii][jj]) * (-basis[i][ii][jj])
            '''
            dres[i] = np.sum(2 * (target - tmp_y) * (-basis[i]))
        # print(dres)
        # print(res)
        res -= lr * dres
        # print(target - sum_all(res, basis))
        loss = np.sum(np.square(target - sum_all(res, basis)))
        # print(loss)
        if (loss < 1e-5):
            break
        j += 1
    return res


# In[ ]:


def get_effs(basis, target):
    A = np.zeros((9,9), dtype = np.float32)
    for i in range(9):
        A[i] = basis[i].reshape(9,)
    return np.matmul(target.reshape(9,), np.linalg.inv(A))


# In[ ]:


# print(conv1w[:,:,0,1] - sum_all(get_efficients(basis, conv1w[:,:,0,1]), basis))


# In[ ]:


for i in range(32):
    a_s[i] = get_efficients(basis, conv1w[:,:,0,i])


# In[ ]:


# print(conv1w[:,:,0,1] - sum_all(a_s[1], basis))


# In[ ]:


# reconstruct the network
input_layer = keras.Input(shape=input_shape) # begin to split the first layer into two layers
l = Conv2D(num_basis, kernel_size=(3, 3))(input_layer) # be careful that the activation should be linear rather than relu
x = Conv2D(32, kernel_size=(1, 1), activation='relu')(l)
for i in range(1, 8):
    x = model.layers[i](x)


# In[ ]:


new_model = keras.Model(input=input_layer, output=x)
# new_model.summary()
new_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[ ]:


conv1_1w, conv1_1b = new_model.layers[1].get_weights()


# In[ ]:


for i in range(num_basis):
    conv1_1w[:,:,0,i] = basis[i]


# In[ ]:


# conv1_1b has been set to 0 initailly
new_model.layers[1].set_weights([conv1_1w, conv1_1b])


# In[ ]:


conv1_2w, conv1_2b = new_model.layers[2].get_weights()


# In[ ]:


for i in range(32):
    for j in range(num_basis):
        conv1_2w[:,:,j,i] = a_s[i][j]


# In[ ]:


new_model.layers[2].set_weights([conv1_2w, conv1b])


# In[ ]:


score = new_model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
print("After weight reconstruction")
print('Test accuracy:', score[1])


# In[ ]:





# In[ ]:




