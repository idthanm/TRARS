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

if __name__ == '__main__':
    test()