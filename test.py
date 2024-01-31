import tensorflow as tf


def test_norm_l1(model):
    # Test if L1 projection of last layer worked
    weights = tf.reshape(model.trainable_variables[-1], [-1])
    norm = tf.norm(weights, ord=1)
    print("norm: {}, gamma: {}".format(norm, model.gamma))


def test_norm_l2(model):
    # Test if L2 projection of inner weights worked
    current_weights = tf.concat(
        [tf.reshape(v, [-1]) for v in model.trainable_weights[:-1]], axis=0
    )
    sub_nets_init_weights = model.sub_nets_init_weights
    diff = sub_nets_init_weights - current_weights
    norm = tf.norm(diff)
    print("norm: {}, delta: {}".format(norm, model.delta))
