
# coding: utf-8

# In[1]:


from keras import layers
from keras.models import Model, Sequential


# In[2]:


def res_block(y, nb_channels, _strides = (1,1), _project_shortcut=False):
        shortcut = y
        
        y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        
        y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
        y = layers.BatchNormalization()()
        
        if _project_shortcut or _strides != (1, 1):
            shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
            
        y = layers.add([shortcut, y])
        y = layers.LeakyReLU()(y)
        
        return y


# In[3]:


def conv_net(x, nb_channels, _strides=(1, 1)):
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=_strides, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=_strides, padding='same', activation='relu')(x)
    
    return x


# In[4]:


def post_net(y, nb_channels, _strides=(1, 1)):
    y = layers.Conv2D(64, kernel_size=(3, 3), strides=_strides, padding='same', activation='relu')(y)
    y = layers.Conv2D(32, kernel_size=(3, 3), strides=_strides, padding='same', activation='relu')(y)
    y = layers.Conv2D(3, kernel_size=(3, 3), strides=_strides, padding='same', activation='linear')(y)
    
    return y


# In[5]:


#inport training data
import numpy as np
x1_train = np.random.random([100,64,64,3])
x2_train = np.random.random([100,64,64,3])
y_train = np.random.random([100, 64, 64, 3])
x1_test = np.random.random([100,64,64,3])
x2_test = np.random.random([100,64,64,3])
y_test = np.random.random([100, 64, 64, 3])

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


# In[6]:


img_a = layers.Input(shape=(64, 64, 3))
img_b = layers.Input(shape=(64, 64, 3))
feature_a = conv_net(img_a, 3)
feature_b = conv_net(img_b, 3)
merge = layers.concatenate([feature_a, feature_b])
aif = post_net(merge, 128)
gen = Model(inputs = [img_a, img_b], output = [aif])
gen.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
gen.summary()


# In[7]:


image_fake = gen([img_a, img_b])
dis = Sequential()
dis.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
dis.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
dis.add(layers.Flatten())
dis.add(layers.Dense(1))
dis.add(layers.Activation('sigmoid'))
pred_prob = dis(image_fake)
dis.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
dis.summary()
make_trainable(dis, False)


# In[8]:


am = Model(inputs = [img_a, img_b], output = [pred_prob])
am.summary()
am.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[9]:


gen.fit([x1_train, x2_train], y_train)
img_fake = gen.predict([x1_train, x2_train])


# In[ ]:


# Train discriminator on generated images
X = np.concatenate((y_train, img_fake))
y = np.zeros([200,2])
y[0:100,1] = 1
y[100:,0] = 1

make_trainable(discriminator,True)
dis.fit(X, y)

# train Generator-Discriminator stack on input noise to non-generated output class
make_trainable(discriminator,False)
am.fit([x1_train, x2_train], y_train)

