import tensorflow as tf
tf.enable_eager_execution()


def test():
    n = 3
    m = 2
    s = tf.Variable(tf.random_normal((n, 1)), dtype=tf.float32, name='state')
    weights = tf.Variable(tf.random_normal((m, n)), dtype=tf.float32, name='weighs')
    with tf.GradientTape() as tape:
        a = tf.matmul(weights, s)

    grad = tape.gradient(a, weights)
    print(grad[:])

def simple_test():
    n = 3
    m = 2
    s = tf.Variable(tf.random_normal((10, n, 1)), dtype=tf.float32, name='state')
    weights = tf.Variable(tf.random_normal((m, n)), dtype=tf.float32, name='weighs')
    actions = tf.matmul(weights, s)
    print(actions.shape)

def test1(ob_size, action_size, feed_dict):
    sess = tf.Session()
    state = tf.placeholder(shape=(ob_size,), dtype=tf.float32, name='state')
    weights = tf.placeholder(shape=(action_size, ob_size), dtype=tf.float32, name='weights')
    actions = tf.matmul(weights, state)

def test2():
    n = 3
    m = 2
    weights = tf.Variable(tf.random.normal((m, n)), dtype=tf.float32, name='weights')
    s = tf.constant([[5], [6], [7]], dtype=tf.float32, name='state')
    with tf.GradientTape() as tape:
        tape.watch(s)
        action = tf.matmul(weights, s)
        print(action)

    print(tape.jacobian(action, weights))

def test_reshape():
    a = tf.Variable([1, 2, 3, 4, 5, 6])
    b = tf.reshape(a, [2, 3])
    c = tf.reshape(a, [3, 2])
    print(a, b, c)

def test_numpy():
    import numpy as np
    a = np.array([1.7, 1.5, 1.6], dtype=np.float32)
    print(a, a.shape)
    a = np.reshape(a, (1, 3))
    print(a, a.shape)
    print(a.transpose())
    b = np.matmul(a.transpose(), a)
    inv_b = np.linalg.pinv(b)
    print(b)
    print(inv_b)
    c = np.random.random((3, 1))
    print(np.matmul(inv_b, c))

def test_singular():
    import numpy as np
    n = 3
    sum_mat = np.zeros((n, n))
    for i in range(10):
        a = np.random.random((1, n))
        mat = np.matmul(a.transpose(), a)
        sum_mat += mat
    inv_sum = np.linalg.inv(sum_mat)
    pinv_sum = np.linalg.pinv(sum_mat)
    rank = np.linalg.matrix_rank(sum_mat)
    print(inv_sum)
    print(pinv_sum)
    print(rank)

if __name__ == '__main__':
    test_singular()