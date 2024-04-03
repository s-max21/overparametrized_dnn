# Import necessary librarys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Input, Dense, Concatenate
from tensorflow.keras.initializers import RandomUniform, Zeros


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
        super().__init__()
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
        super().__init__()
        # Error handling:
        if gamma is None:
            raise ValueError("Missing required argument gamma.")
        self.gamma = tf.constant(gamma, dtype=tf.float32)

    @tf.function
    def __call__(self, w):
        return self.apply_l1_projection(w)

    @tf.function
    def apply_l1_projection(self, w):
        if tf.norm(w, ord=1) <= self.gamma:
            return w
        # Apply L1-projection on weight vector w
        abs_w = tf.abs(w)

        # Compute cumulative sum of the sorted absolute weights
        u = tf.sort(abs_w, direction="DESCENDING", axis=0)
        svp = tf.cumsum(u, axis=0) - self.gamma

        # Find the position where the condition is violated for the first time
        ratio = svp / tf.reshape(
            tf.range(1, tf.size(u) + 1, dtype=tf.float32), shape=tf.shape(w)
        )

        # Compute the threshold value
        theta = tf.reduce_max(tf.maximum(ratio, 0.0), axis=0)

        return tf.math.sign(w) * tf.maximum(abs_w - theta, 0)

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
        The value to specify radius of the L2-ball around 'init_vars'.
        The weights will be projected on the feasible set where the L2-norm of
        the difference between current weight vector and the initial weight
        vector is smaller or equal to delta.

    """

    def __init__(
        self,
        delta=1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.delta = tf.constant(delta, dtype=tf.float32)
        self.init_vars = None

    @tf.function(reduce_retracing=True)
    def train_step(self, data):
        """
        Custom training step for the model.

        After applying the gradients, the weight vector will be projected on the
        L2-ball with radius 'delta' around the center 'init_vars'.

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

        # Apply L2 projection
        vars_nets = self.trainable_variables[:-1]
        vars_diff = [v - v0 for v, v0 in zip(vars_nets, self.init_vars)]

        if tf.linalg.global_norm(vars_diff) > self.delta:
            # Project the weights and reshape back
            projected_vars = self.apply_l2_projection(vars_diff)
            for v, pv in zip(vars_nets, projected_vars):
                v.assign(pv)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def apply_l2_projection(self, vars_diff):
        projected_diff = tf.clip_by_global_norm(vars_diff, self.delta)[0]
        return [v0 + pv for v0, pv in zip(self.init_vars, projected_diff)]

    def get_config(self):
        return {
            "delta": self.delta,
            "init_vars": self.init_vars,
        }


def create_network(n=100, num_neurons=10, num_layers=10, beta=10):
    """
    Creates a neural network with num_layers hidden layers and a truncation layer as
    the last layer.

    Parameters
    ----------
    n: int
        number of training samples
    num_neurons: int, optional
        number of neurons in each hidden layer
    num_layers: int, optional
        number of hidden layers
    beta: float, optional
        parameter for the truncation layer

    Returns
    -------
    model: keras.models.Sequential
        Sequential model containing a truncation layer as last layer

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
    for _ in range(num_layers - 1):
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

    # Create truncation layer
    model.add(TruncateLayer(beta=beta))

    return model


def create_dnn(
    train_shape,
    num_networks=100,
    num_layers=10,
    num_neurons=10,
    beta=10,
    gamma=10,
    delta=1,
):
    """
    Creates a model with num_networks subnetworks, each consisting of num_layers hidden layers.
    The output is the output of the subnetworks combined in a last dense layer.

    Parameters
    ----------
    train_shape: tuple
        Shape of the training data.
    num_networks: int, optional
        Number of subnetworks to train.
    num_layers: int, optional
        Number of hidden layers in each subnetwork.
    num_neurons: int, optional
        Number of neurons in each hidden layer.
    beta: float, optional
        Parameter for the truncation layer.
    gamma: float, optional
        Parameter for the L1 projection layer.
    delta: float, optional
        Parameter for the L2 projection.

    Returns
    -------
    model: L2ProjectionModel
        The created model with the specified structure.
    """

    # Define input shape based on dimension of input variable
    n, d = train_shape
    input_shape = (d,)

    # Create a list containing num_networks DNNs with num_layers hidden layers
    sub_networks = [
        create_network(n=n, num_neurons=num_neurons, num_layers=num_layers, beta=beta)
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
    )

    # One-time initialization of initial weights
    model.init_vars = [tf.identity(v) for v in model.trainable_variables[:-1]]

    return model
