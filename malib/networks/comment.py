# Created by yingwen at 2019-03-10

import tensorflow as tf
from malib.utils.keras import PicklableKerasModel


def CommNet(
    input_shapes,
    output_size,
    hidden_layer_sizes,
    activation="relu",
    output_activation="linear",
    preprocessors=None,
    name="commnet",
    *args,
    **kwargs
):
    inputs = [tf.keras.layers.Input(shape=input_shape) for input_shape in input_shapes]
    n = input_shapes[0][0]
    if preprocessors is None:
        preprocessors = (None,) * len(inputs)

    preprocessed_inputs = [
        preprocessor(input_) if preprocessor is not None else input_
        for preprocessor, input_ in zip(preprocessors, inputs)
    ]

    concatenated = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(
        preprocessed_inputs
    )

    out = concatenated
    for units in hidden_layer_sizes:
        out = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units, *args, activation=output_activation, **kwargs)
        )(out)
        mean_out = tf.reduce_mean(out, axis=1)
        mean_out = tf.keras.backend.repeat(mean_out, n)
        out = tf.concat((out, mean_out), axis=-1)

    out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            output_size, *args, activation=output_activation, **kwargs
        )
    )(out)

    model = PicklableKerasModel(inputs, out, name=name)

    return model
