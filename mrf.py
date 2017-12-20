import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.image import transform
# # # # # Offset image fcn
def translate(images, tx, ty, interpolation='NEAREST'):
    transforms = [1, 0, -tx, 0, 1, -ty, 0, 0]
    return transform(images, transforms, interpolation)

img = cv2.imread('restore.png') # <---- image to restore
useCPUonly = True
if (useCPUonly):
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def mgrad(img, L = 0.1, LR = 1, initial = 'same', steps = 10, toFile = False):
    img = np.array(img, dtype = np.float32)
    if (initial == 'mean'):
        img_initial = img.mean()*np.ones(np.shape(img), dtype = np.float32)
    elif (initial == 'zeros'):
        img_initial = np.zeros(np.shape(img), dtype = np.float32)
    else:
        img_initial = img
    #  Create Graph
    orig = tf.constant(img)
    
    x = tf.Variable(img_initial)
    x1 = translate(x, tx=-1, ty=0)
    x2 = translate(x, tx= 1, ty=0)
    x3 = translate(x, tx=0, ty=-1)
    x4 = translate(x, tx=0, ty= 1)
    m = 250.0**2 # <- Configure manually
    energy = (L*tf.squared_difference(orig,x)+(1-L)*(\
	tf.minimum( m, tf.squared_difference(x1,x) )+\
	tf.minimum( m, tf.squared_difference(x2,x) )+\
	tf.minimum( m, tf.squared_difference(x3,x) )+\
	tf.minimum( m, tf.squared_difference(x4,x) ) \
    ))
    orig = x
    # # # # #  Configure descend rate
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = LR
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1, 0.9, staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(LR)
    # # # # # Perform Δescend
    train = optimizer.minimize(energy, global_step=global_step)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(steps):  
            sess.run(train)
            # if ((step+1==steps) and (toFile)):
            cv2.imwrite('test/'+str(step)+'out.png',sess.run(x) )	
        return sess.run(x)
mgrad(img, L=0.1, LR = 0.1, initial = 'same', toFile = True, steps = 20)    

input('Finished. Press CTRL+C to return.')