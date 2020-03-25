# Created by yingwen at 2019-03-10

import tensorflow as tf
from malib.utils.keras import PicklableKerasModel


def BiCNet(
    input_shapes,
    output_size,
    hidden_layer_sizes,
    activation="relu",
    output_activation="linear",
    preprocessors=None,
    name="bicnet",
    *args,
    **kwargs
):
    inputs = [tf.keras.layers.Input(shape=input_shape) for input_shape in input_shapes]

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
        out = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, return_sequences=True)
        )(out)

    out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            output_size, *args, activation=output_activation, **kwargs
        )
    )(out)

    model = PicklableKerasModel(inputs, out, name=name)

    return model