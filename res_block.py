
# coding: utf-8

# In[36]:


from keras import layers
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras import backend as K
#from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython import display


# In[37]:


def res_block(y, nb_channels, _strides = (1,1), _project_shortcut=False):
    shortcut = y

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    #y = layers.BatchNormalization()()

    if _project_shortcut or _strides != (1, 1):
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    #y = layers.LeakyReLU()(y)

    return y


# In[38]:


def res_net(x, nb_channels, _strides=(1, 1)):
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=_strides, padding='same', activation='relu')(x)
    shortcut = x
    for _ in range(16):
        x = res_block(x, 64)

    x = layers.Conv2D(64, kernel_size=(3, 3), strides=_strides, padding='same', activation='relu')(x)
    x = layers.add([shortcut, x])
    
    return x


# In[39]:


def conv_net(x, nb_channels, _strides=(1, 1)):
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=_strides, padding='same', activation='relu')(x)
    #x = layers.Conv2D(64, kernel_size=(3, 3), strides=_strides, padding='same', activation='relu')(x)
    
    return x


# In[40]:


def post_net(y, nb_channels, _strides=(1, 1)):
    #y = layers.Conv2D(64, kernel_size=(3, 3), strides=_strides, padding='same', activation='relu')(y)
    #y = layers.Conv2D(32, kernel_size=(3, 3), strides=_strides, padding='same', activation='relu')(y)
    y = layers.Conv2D(3, kernel_size=(3, 3), strides=_strides, padding='same', activation='linear')(y)
    
    return y


# In[41]:


import cv2
import numpy as np

def load_imgs(path, number, train_type):
    result=np.empty((number, 64, 64, 3), dtype="float64")
    for i in range(number):
        I = cv2.imread(path + "{:04}_{}.jpeg".format(i+1, train_type))
        result[i, :, :, :] = I
    return result/result.max()


# In[42]:


#inport training data
dataNum = 1000
x1_train = load_imgs("./blurImg/", dataNum, 1)
x2_train = load_imgs("./blurImg/", dataNum, 2)
y_train = load_imgs("./blurImg/", dataNum, 0)

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
        
def loss_wrapper(in_tensor1, in_tensor2):
    def gaussian_blur(in_tensor):
        # use large kernel to blur pred and in_tensor//
        return
        
    def custom_loss(y_true, y_pred):
        # or better implementation like fourier transformation
        return K.binary_crossentropy(y_true, y_pred) + K.reduce_mean(K.square(gaussian_blur(y_pred)-gaussian_blur(in_tensor1)))
    return custom_loss


# In[43]:


img_a = layers.Input(shape=(64, 64, 3))
img_b = layers.Input(shape=(64, 64, 3))
#feature_a = conv_net(img_a, 3)
#feature_b = conv_net(img_b, 3)
feature_a = res_net(img_a, 3)
feature_b = res_net(img_b, 3)
merge = layers.concatenate([feature_a, feature_b])
aif = post_net(merge, 128)
gen = Model(inputs = [img_a, img_b], output = [aif])
gen.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
gen.summary()
#plot_model(gen, to_file='generator.png')


# In[46]:


Batch_size = 16
nb_epoch = 100
for epoch in range(nb_epoch):
    rand_idx = np.random.randint(0, x1_train.shape[0], size = Batch_size)
    img_batch1 = x1_train[rand_idx, :, :, :]
    img_batch2 = x2_train[rand_idx, :, :, :]
    y_batch = y_train[rand_idx, :, :, :]
    gen.fit([img_batch1, img_batch2], y_batch)

gen_img = gen.predict([img_batch1, img_batch2])


# In[8]:

'''
image_fake = gen([img_a, img_b])
dis = Sequential()
dis.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same'))
dis.add(layers.LeakyReLU())
dis.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
dis.add(layers.LeakyReLU())
dis.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same'))
dis.add(layers.LeakyReLU())
dis.add(layers.Conv2D(1, kernel_size=(3, 3), padding='same'))

dis.add(layers.Flatten())
dis.add(layers.Dense(512, activation='tanh'))
dis.add(layers.Dense(1))
dis.add(layers.Activation('sigmoid'))
pred_prob = dis(image_fake)
dis.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
dis.summary()
plot_model(dis, to_file='discriminator.png')
make_trainable(dis, False)


# In[9]:


am = Model(inputs = [img_a, img_b], output = [pred_prob])
am.summary()
am.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
plot_model(am, to_file='adversary.png')


# In[10]:


#pre-train discriminate network


# In[11]:


def plot_loss(losses):
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.figure(figsize=(10,8))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g"], label='generative loss')
        plt.legend()
        plt.show()


# In[12]:


# Train discriminator on generated images
losses = {"d":[], "g":[]}
Batch_size = 16
nb_epoch = 100
for epoch in tqdm(range(nb_epoch)):
    rand_idx = np.random.randint(0, x1_train.shape[0], size = Batch_size)
    img_batch1 = x1_train[rand_idx, :, :, :]
    img_batch2 = x2_train[rand_idx, :, :, :]
    y_batch = y_train[np.random.randint(0, y_train.shape[0], size = Batch_size), :, :, :]
    #gen.fit([x1_train, x2_train], y_train)
    gen_img = gen.predict([img_batch1, img_batch2])
    X = np.concatenate((y_batch, gen_img))
    y = np.zeros([2*Batch_size,])
    y[0:Batch_size] = 1
    y[Batch_size:] = 0
    make_trainable(dis,True)
    d_loss = dis.train_on_batch(X, y)
    losses["d"].append(d_loss)
    
    y2 = np.ones([Batch_size, ])
    # train Generator-Discriminator stack on input noise to non-generated output class
    make_trainable(dis,False)
    g_loss = am.train_on_batch([img_batch1, img_batch1], y2) #same batch or ???
    losses["g"].append(g_loss)
    if epoch % 25 == 25 - 1:
        plot_loss(losses)
'''
