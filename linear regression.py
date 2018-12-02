import numpy as np
import tensorflow as tf


def linear_reg():
    n_samples, batch_size, num_steps = 1000, 100, 10000
    x_data = np.random.uniform(1, 10, (n_samples, 1))
    eps = np.random.normal(0, 2, (n_samples, 1))
    y_data = 2 * x_data + 1 + eps
    x = tf.placeholder(tf.float32, shape=(batch_size, 1))
    y = tf.placeholder(tf.float32, shape=(batch_size, 1))

    with tf.variable_scope('linear-regression'):
        k = tf.Variable(tf.random_normal((1, 1), stddev=0.0), name='slope')
        b = tf.Variable(tf.zeros((1,)), name='bias')

    y_pred = tf.matmul(x, k) + b
    loss = tf.reduce_mean((y - y_pred) ** 2)
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    display_step = 50

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_steps):
            indices = np.random.choice(n_samples, size=batch_size)
            x_batch, y_batch = x_data[indices], y_data[indices]
            _, loss_val, k_val, b_val = sess.run([optimizer, loss, k, b],
                                                 feed_dict={x: x_batch, y: y_batch})
            if (i + 1) % display_step == 0:
                print('Epoch %d: %.8f, k=%.4f, b=%.4f' % (i + 1, loss_val, k_val, b_val))


linear_reg()
# Create 1000 random points in [0:1]
# Calculate for each point x the correct answer y = 2x+1+ϵ, ϵ  N(ϵ; 0; 2- variance)
# Initialization of k and b
# Calculating the initial loss function
# Calculating the optimizer, learning rate = 0.001
# Optimization cycle