from utils import *

## Define Placeholders
input_placeholder = tf.placeholder(tf.float32, shape=[None,512,512,3], name="inputs")
label_placeholder = tf.placeholder(tf.float32, shape=[None,512,512,3], name="labels")

## Define variables
with tf.variable_scope("model_vars"):
    w1 = tf.get_variable("w1", [3,3,3,16], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False), trainable=True)
    w2 = tf.get_variable("w2", [3,3,16,3],initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False), trainable=True)

    b1 = tf.get_variable("b1", [16], initializer=tf.constant_initializer(0), trainable=True)
    b2 = tf.get_variable("b2", [3],  initializer=tf.constant_initializer(0), trainable=True)

## Define model (& layers)
def batch_norm(input_tensor, is_training, name):
    decay = 0.9
    epsilon = 1e-5
    shape = input_tensor.get_shape()[-1]
    with tf.variable_scope("model_vars/"+name, reuse = is_training==False):
        offset  = tf.get_variable(name="Beta",  shape=shape, initializer=tf.constant_initializer(0.0), trainable=True)
        scale   = tf.get_variable(name="Gamma", shape=shape, initializer=tf.constant_initializer(1.0), trainable=True)
        moving_mean = tf.get_variable(name="mu",    shape=shape, initializer=tf.constant_initializer(0.0), trainable=False) #To be calculate. Not to be trained
        moving_var  = tf.get_variable(name="sigma", shape=shape, initializer=tf.constant_initializer(1.0), trainable=False) #To be calculate. Not to be trained

    if is_training==True:
        print("[BN] Training time")
        batch_mean, batch_var = tf.nn.moments(input_tensor, [0, 1, 2])
        train_mv_mean = tf.assign(moving_mean, moving_mean * decay + batch_mean * (1 - decay))
        train_mv_var  = tf.assign(moving_var,  moving_var  * decay + batch_var  * (1 - decay))
        # Add control dependency to update Mu & Sigma values. Else values will remain as initialized - 0 & 1.
        with tf.control_dependencies([train_mv_mean, train_mv_var]):
            output_tensor = tf.nn.batch_normalization(input_tensor, batch_mean, batch_var, offset, scale, epsilon)
    else:
        print("[BN] Test time ")
        output_tensor = tf.nn.batch_normalization(input_tensor, moving_mean, moving_var, offset, scale, epsilon)

    return output_tensor

def model(is_training):
    conv1   = tf.nn.conv2d(input_placeholder, w1, strides=[1,1,1,1], padding="SAME")
    conv1_b = tf.nn.bias_add(conv1, b1)
    bn1     = batch_norm(conv1_b, is_training, "BN-1")

    conv2   = tf.nn.conv2d(bn1, w2, strides=[1,1,1,1], padding="SAME")
    conv2_b = tf.nn.bias_add(conv2, b2)
    bn2     = batch_norm(conv2_b, is_training, "BN-2")

    output = tf.add(bn2, input_placeholder, name="output")
    return output

## Dataset
input_img = get_lena()
tr_input, tr_label = get_tr_dataset()
te_input = get_te_dataset()

## Loss & Optimizer
output_tr = model(True)
output_te = model(False)
loss = tf.losses.mean_squared_error(output_tr, label_placeholder)
train_op = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

trainable_vars = get_trainable_vars()
global_vars = get_global_vars()
mu_sigma_list = [var for var in global_vars if ("mu" in var.name or "sigma" in var.name)]
print_list(global_vars, "Global Variables")
print_list(trainable_vars, "All Trainable Variables")
print_list(mu_sigma_list, "Mu-Sigma")

with tf.Session() as sess:
    sess.run(init) #Do Not put init after restoring checkpoints

    # saver.restore(sess, tf.train.latest_checkpoint("checkpoint/"))
    # print("Model restored.")

    out_img = sess.run(output_te, feed_dict={input_placeholder:input_img})
    save_model_out(out_img, "before")
    
    start = timer()
    for epoch in range(5):
        _, loss_val = sess.run([train_op, loss], feed_dict={
            input_placeholder:tr_input,
            label_placeholder:tr_label
        })
        print("Epcoh: %2d| Loss: %f" %(epoch, loss_val))

        '''Without control dependency, mu & sigma values won't update'''
        mu_sigma_vals = sess.run(mu_sigma_list)
        print_list(mu_sigma_vals, "Mu-Sigma Values")

    end = timer()
    print("==> Time to train: {t} sec".format(t=end-start))

    # save_path = saver.save(sess, "checkpoint/model-500.ckpt")
    # print("Model saved in path: %s" % save_path)
    
    out_img = sess.run(output_te, feed_dict={input_placeholder:input_img})
    save_model_out(out_img, "after")
    out_img = sess.run(output_te, feed_dict={input_placeholder:te_input})
    save_model_out(out_img[0], "test")
