import numpy as np
import tensorflow as tf

#pathに保存されているMNISTデータを読み込む
def Load_MNIST_DATA_as_Numpy(path):

    X_train = np.load(path + 'mnist_X_train.npy')
    y_train = np.load(path + 'mnist_y_train.npy')
    X_test = np.load(path + 'mnist_X_test.npy')
    y_test = np.load(path + 'mnist_y_test.npy')

    return (X_train, y_train, X_test, y_test)

# モデルクラス
class Mnist_Model(object):
    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.X = tf.placeholder(tf.uint8, [None, 28, 28, 1])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.l = tf.placeholder(tf.float32, [])

        filter_count = 8

        conv_1 = tf.keras.layers.Conv2D(filter_count, 3, 1)(self.X)
        pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv_1)

        conv_2 = tf.keras.layers.Conv2D(filter_count * 2, 3, 1)(pool_1)
        pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv_2)

        fc1 = tf.keras.layers.Dense(100)(pool_2)
        fc1_relu = tf.keras.activations.relu(fc1)

        fc2 = tf.keras.layers.Dense(100)(fc1_relu)

        self.pred = tf.keras.activations.softmax(fc2)
        self.loss = tf.keras.losses.categorical_crossentropy(self.y , self.pred)



def Training(graph, model, Data, num_steps,verbose):
    # Build the model graph
    graph = tf.get_default_graph()
    with graph.as_default():
        model = MNISTModel()

        learning_rate = tf.placeholder(tf.float32, [])
        pred_loss = tf.reduce_mean(model.pred_loss)
        train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)

        # Evaluation
        correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
        label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))


    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        # Training loop
        for i in range(num_steps):
            # Adaptation param and learning rate schedule as described in the paper
            p = float(i) / num_steps
            l = 2. / (1. + np.exp(-10. * p)) - 1
            lr = 0.01 / (1. + 10 * p)**0.75

            # Training step
            X0, y0 = next(gen_source_batch)
            X1, y1 = next(gen_target_batch)
            X = np.vstack([X0, X1])
            y = np.vstack([y0, y1])

            _, batch_loss, ploss, p_acc = sess.run(
                [train_op, pred_loss, label_acc],
                feed_dict={model.X: X, model.y: y, model.l: l, learning_rate: lr})

            if verbose and i % 100 == 0:
                print('loss: {}   p_acc: {}  p: {}  l: {}  lr: {}'.format(batch_loss, p_acc, p, l, lr))

        # Compute final evaluation on test data
        source_acc = sess.run(label_acc,
                            feed_dict={model.X: mnist_test, model.y: mnist.test.labels,
                                       model.train: False})

        target_acc = sess.run(label_acc,
                            feed_dict={model.X: mnistm_test, model.y: mnist.test.labels,
                                       model.train: False})
