from __future__ import print_function
import os
import shutil
import numpy as np
import tensorflow as tf
import ebm_pca_data as d
from six.moves import xrange
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

Y_DIM = 2
Z_DIM = 1
BATCH_SIZE = 1000
LEARNING_RATE = 1e-3 
ITER_LIMIT = 1000
LOG_DIR = '/Users/rogovski/.clatter/models/demo/ebm_pca/'

if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
    os.mkdir(LOG_DIR)

with tf.variable_scope("observed_variables"):
    y = tf.placeholder("float", [None, Y_DIM])

with tf.variable_scope("latent_variables"):
    z = tf.placeholder("float", [None, Z_DIM])

with tf.variable_scope("trainable_params"):
    W = tf.Variable(
        tf.random_normal([Z_DIM, Y_DIM], stddev=0.02), 
        name="weights"
    )
    tf.summary.histogram("W", W)

with tf.variable_scope("encoder"):
    enc_y = tf.matmul(W,tf.transpose(y))

with tf.variable_scope("decoder"):
    dec_z = tf.matmul(tf.transpose(W),tf.transpose(z))

with tf.variable_scope("training"):
    #loss_enc = tf.square(tf.norm(enc_y - z), name="encoder_loss")
    #loss_dec = tf.square(tf.norm(dec_z - tf.transpose(y)), name="decoder_loss")
    #loss = tf.add(loss_enc, loss_dec, name="energy")
    loss = tf.square(tf.norm( \
            tf.matmul(tf.transpose(W), tf.matmul(W, tf.transpose(y))) - tf.transpose(y)), 
            name="energy")
    tf.summary.scalar("energy", loss)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    grads = optimizer.compute_gradients(loss, var_list=[W])
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradient", grad)
    train_op = optimizer.apply_gradients(grads)

# with tf.variable_scope("free_energy"):
#     z = tf.placeholder("float", [None, Z_DIM])

print("Setting up summary op")
summary_op = tf.summary.merge_all()

ds = d.DataReader()
dataset_train = d.BatchDataset(ds.y,ds.z)

def free_energy(Wres, Zspace):
    lenz = Zspace.shape[0]
    def go(y0,y1):
        point_y0 = np.repeat(y0, lenz)
        point_y1 = np.repeat(y1, lenz)
        grid_point = np.array(zip(point_y0, point_y1))
        enc_res = np.dot(Wres, grid_point.T)
        dec_enc_res = np.dot(Wres.T, enc_res)
        return np.linalg.norm(dec_enc_res - grid_point.T)**2
    return go

def free_energy2(Y, Wres):
    enc_res = np.dot(Wres, Y.T)
    dec_enc_res = np.dot(Wres.T, enc_res)
    return np.linalg.norm(dec_enc_res - Y.T)**2

with tf.Session() as sess:
    print("Setting up Saver")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(LOG_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored")

    for it in xrange(ITER_LIMIT):
        _y_train_batch, _z_train_batch = dataset_train.next_batch(BATCH_SIZE)
        feed_dict = { y: _y_train_batch }
        sess.run(train_op, feed_dict=feed_dict)

        # print the loss every 50 iterations
        if it % 10 == 0:
            train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
            print("Step: {}, Train_loss: {}".format(it, train_loss))
            summary_writer.add_summary(summary_str, it)

    Wres = sess.run(W)
    z_space = np.linspace(0, 1, 1000).reshape(-1,1)
    y0_space = np.linspace(-30, 30, 50)
    y1_space = np.linspace(-30, 30, 50)
    y0_space, y1_space = np.meshgrid(y0_space, y1_space)
    y0_space = y0_space.reshape(-1)
    y1_space = y1_space.reshape(-1)
    e_surface = []
    for i in range(len(y0_space)):
        e_surface_point = free_energy2(np.array([[y0_space[i], y1_space[i]]]), Wres)
        e_surface.append(e_surface_point)
    e_surface = np.array(e_surface)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(y0_space.reshape(-1,50), y1_space.reshape(-1,50), e_surface.reshape(-1,50), cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)
    plt.show()
