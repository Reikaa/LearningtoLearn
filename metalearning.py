import tensorflow as tf

DIMS = 10  # Dimensions of the parabola
scale = tf.random_uniform([DIMS], 0.5, 1.5)
# This represents the network we are trying to optimize,
# the `optimizee' as it's called in the paper.
# Actually, it's more accurate to think of this as the error
# landscape.
def f(x):
    x = scale*x
    return tf.reduce_sum(x*x)
    
def g_sgd(gradients, state, learning_rate=0.1):
    return -learning_rate*gradients, state
    
def g_rms(gradients, state, learning_rate=0.1, decay_rate=0.99):
    if state is None:
        state = tf.zeros(DIMS)
    state = decay_rate*state + (1-decay_rate)*tf.pow(gradients, 2)
    update = -learning_rate*gradients / (tf.sqrt(state)+1e-5)
    return update, state
    
    
TRAINING_STEPS = 20  # This is 100 in the paper

initial_pos = tf.random_uniform([DIMS], -1., 1.)

def learn(optimizer):
    losses = []
    x = initial_pos
    state = None
    for _ in range(TRAINING_STEPS):
        loss = f(x)
        losses.append(loss)
        grads, = tf.gradients(loss, x)
      
        update, state = optimizer(grads, state)
        x += update
    return losses
    
    
    
sgd_losses = learn(g_sgd)
rms_losses = learn(g_rms)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(TRAINING_STEPS)
for _ in range(3): 
    sgd_l, rms_l = sess.run([sgd_losses, rms_losses])
    p1, = plt.plot(x, sgd_l, label='SGD')
    p2, = plt.plot(x, rms_l, label='RMS')
    plt.legend(handles=[p1, p2])
    plt.title('Losses')
    plt.show()


