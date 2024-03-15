
# Import necessary librarys
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Input, Dense, Concatenate
from keras.initializers import RandomUniform, Zeros



class TruncateLayer(Layer):
    """
    A Keras layer that truncates every input to a specified range.

    Parameters
    ----------
    beta : float
        The value used to determine the truncation range.
        The inputs will be clipped to the range [-beta, beta].

    Methods
    -------
    call(inputs)
        Applies truncation to the inputs based on the `beta` value.

    """

    def __init__(self, beta):
        super(TruncateLayer, self).__init__()
        self.beta = tf.constant(beta, dtype=tf.float32)

    def call(self, inputs):
        return tf.clip_by_value(
            inputs, clip_value_min=-self.beta, clip_value_max=self.beta
        )


class L1Projection(keras.constraints.Constraint):
    """
    A Keras weight constraint that projects the weight vector w on a vector
    with L1-norm smaller or equal to gamma.

    Parameters
    ----------
    gamma: float
        The value to constraint the L1-norm.
        The weights will be projected on a feasible set where the L1-norm of
        the weight vector is smaller or equal to gamma.

    Methods
    -------
    projection_l1(w)
        Applies L1-projection to the weight vector w based on the 'gamma' value.

    """

    def __init__(self, gamma):
        super(L1Projection, self).__init__()
        self.gamma = tf.constant(gamma, dtype=tf.float32)

        # Error handling:
        if self.gamma is None:
            raise ValueError("Missing required argument gamma.")

    def __call__(self, w):
        return self.apply_l1_projection(w)

    @tf.function(reduce_retracing=True)
    def apply_l1_projection(self, w):
        # Test if projection necessary
        if tf.norm(w, ord=1) <= self.gamma:
            return w
        else:
            # Apply L1-projection on weight vector w
            w_shape = w.shape
            w_flat = tf.reshape(w, [-1])
            abs_w_flat = tf.abs(w_flat)

            # Compute cumulative sum of the sorted absolute weights
            u = tf.sort(abs_w_flat, direction="DESCENDING")
            svp = tf.cumsum(u)

            # Find the position where the condition is violated for the first time
            cond = tf.cast(svp - self.gamma, tf.float64) / tf.range(
                1, tf.size(u) + 1, dtype=tf.float64
            )
            k = tf.where(tf.cast(u, tf.float64) > cond)[-1][0]

            # Compute the threshold value
            theta = tf.cast(svp[k] - self.gamma, tf.float32) / tf.cast(
                k + 1, tf.float32
            )

            # Apply the thresholding operation
            projected_weights_flat = tf.math.sign(w_flat) * tf.maximum(
                abs_w_flat - theta, 0
            )
            return tf.reshape(projected_weights_flat, shape=w_shape)

    def get_config(self):
        return {"gamma": self.gamma}


class L2ProjectionModel(keras.Model):
    """
    A Keras model that performs an L2-projection of the weights in every
    training step, to ensure that the eucledian distance between the weights and
    the initial weights is smaller or equal to 'delta'.

    Parameters
    ----------
    delta: float
        The value to specify radius of the L2-ball around 'sub_nets_init_weights'.
        The weights will be projected on the feasible set where the L2-norm of
        the difference between current weight vector and the initial weight
        vector is smaller or equal to delta.

    """

    def __init__(
        self,
        delta,
        sub_networks=None,
        output_layer=None,
        num_networks=None,
        num_layers=None,
        num_neurons=None,
        beta=None,
        gamma=None,
        **kwargs
    ):
        super(L2ProjectionModel, self).__init__(**kwargs)
        self.delta = tf.constant(delta, dtype=tf.float32)
        self.sub_nets_init_weights = None

        # Error handling:
        if self.delta is None:
            raise ValueError("Missing required argument delta.")

        # Optional parameters for analysis
        self.sub_networks = sub_networks
        self.output_layer = output_layer
        self.num_networks = num_networks
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.beta = tf.constant(beta)
        self.gamma = tf.constant(gamma)

    @tf.function(reduce_retracing=True)
    def train_step(self, data):
        """
        Custom training step for the model.

        After applying the gradients, the weight vector will be projected on the
        L2-ball with radius 'delta' around the center 'sub_nets_init_weights'.

        Parameters
        ----------
        data: float
            data used for training

        """

        # Unpack the data used for training
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Compute the loss value
            loss = self.compute_loss(y=y, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Get current weights and compute global weight difference
        current_weights = tf.concat(
            [tf.reshape(w, [-1]) for w in self.trainable_variables[:-1]], axis=0
        )
        sub_nets_init_weights = self.sub_nets_init_weights
        weights_diff = current_weights - sub_nets_init_weights

        # Apply L2 projection
        if tf.norm(weights_diff) > self.delta:
            # Project the weights and reshape back
            self.apply_l2_projection(weights_diff)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function(reduce_retracing=True)
    def apply_l2_projection(self, weights_diff):
        """
        Projection on L2-ball with center 'sub_nets_init_weights' and radius 'delta'.

        """

        # Projection of the weights
        projected_weights = self.sub_nets_init_weights + tf.clip_by_norm(
            weights_diff, self.delta
        )

        # Reshape the projected vector to assign weights correctly
        start = 0
        for w in self.trainable_variables[:-1]:
            shape = w.shape
            size = tf.reduce_prod(shape)
            w.assign(tf.reshape(projected_weights[start : start + size], shape))
            start += size

    def get_config(self):
        return {
            "gamma": self.gamma,
            "delta": self.delta,
            "beta": self.beta,
            "num_networks": self.num_networks,
            "num_layers": self.num_layers,
            "num_neurons": self.num_neurons,
        }


def create_sub_network(n=400, num_neurons=5, num_layers=None, beta=None):
    """
    Creates a submodel with num_layers hidden layers and a truncation layer as
    the last layer.

    Parameters
    ----------
    n: int
        number of training samples
    num_neurons: int, optional
        number of neurons in each hidden layer (default: 5)
    num_layers: int, optional
        number of hidden layers (default: math.ceil(np.log(n)))
    beta: float, optional
        parameter for the truncation layer (default: 100*np.log(n))

    Returns
    -------
    model: keras.models.Sequential
        submodel containing a truncation layer as last layer

    """

    # Define submodel
    model = keras.models.Sequential()

    # Create input layer
    model.add(
        Dense(
            units=num_neurons,
            activation="relu",
            kernel_initializer=RandomUniform(minval=-n, maxval=n),
            bias_initializer=RandomUniform(minval=-n, maxval=n),
        )
    )

    # Create num_layers-1 hidden layers
    for _ in range(1, num_layers - 1):
        model.add(
            Dense(
                units=num_neurons,
                activation="relu",
                kernel_initializer=RandomUniform(minval=-1, maxval=1),
                bias_initializer=RandomUniform(minval=-1, maxval=1),
            )
        )

    # Create output layer
    model.add(
        Dense(
            units=1,
            activation="relu",
            kernel_initializer=RandomUniform(minval=-1, maxval=1),
            bias_initializer=RandomUniform(minval=-1, maxval=1),
        )
    )

    # Create tuncation layer
    model.add(TruncateLayer(beta=beta))

    return model


def create_dnn(
    train_shape,
    num_networks=None,
    num_layers=None,
    num_neurons=5,
    beta=None,
    gamma=None,
    delta=1,
):
    """
    Creates a model with num_networks subnetworks with num_layers hidden layers
    each. The output is the average of the outputs of the subnetworks.

    Parameters
    ----------
    train_shape: tuple
        shape of the training data
    num_networks: int, optional
        number of subnetworks to train (default: math.ceil(n**((np.log(n)**1.1 + 1))))
    num_layers: int, optional
        number of hidden layers in each subnetwork (default: math.ceil(np.log(n)))
    num_neurons: int, optional
        number of neurons in each hidden layer (default: 5)
    beta: float, optional
        parameter for the truncation layer (default: 10*np.log(n))
    gamma: float, optional
        parameter for the L1 projection layer (default: 10*n**(d/(2*(1+d))))
    delta: float, optional
        parameter for the L2 projection (default: 1)

    """
    # Define input shape based on dimension of input variable
    n, d = train_shape
    input_shape = (d,)

    # Create a list containing num_networks DNNs with num_layers hidden layers
    sub_networks = [
        create_sub_network(
            n=n, num_neurons=num_neurons, num_layers=num_layers, beta=beta
        )
        for _ in range(num_networks)
    ]

    # Create the output layer
    output_layer = Dense(
        units=1,
        use_bias=False,
        kernel_initializer=Zeros(),
        kernel_constraint=L1Projection(gamma),
    )

    # Define the structure of the combined model
    inputs = Input(shape=input_shape)
    truncated_outputs = [sub_net(inputs) for sub_net in sub_networks]
    concatenated_outputs = Concatenate()(truncated_outputs)
    outputs = output_layer(concatenated_outputs)

    # Create the model
    model = L2ProjectionModel(
        inputs=inputs,
        outputs=outputs,
        delta=delta,
        sub_networks=sub_networks,
        output_layer=output_layer,
        num_networks=num_networks,
        num_layers=num_layers,
        num_neurons=num_neurons,
        beta=beta,
        gamma=gamma,
    )

    # One-time initialization of initial weights
    if model.sub_nets_init_weights is None:
        sub_nets_init_weights = tf.concat(
            [tf.reshape(w, [-1]) for w in model.trainable_variables[:-1]], axis=0
        )
        model.sub_nets_init_weights = sub_nets_init_weights

    return model
