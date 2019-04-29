from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.regularizers import l2
from kerascv.model_provider import get_model as kecv_get_model
from opconty_shufflenetv2 import ShuffleNetV2


def get_mobilenet_v2(weight_decay=0.0005, input_shape=(64, 64, 3)):
    keras_model = MobileNetV2(input_shape=input_shape, include_top=False, weights=None)
    # keras_model.summary()
    # Total params: 2,257,984
    # Trainable params: 2,223,872
    # Non-trainable params: 34,112

    for layer in keras_model.layers:
        layer.trainable = True
    net_layer_len = len(keras_model.layers)
    flatten = Flatten()(keras_model.layers[net_layer_len - 1].output)
    predictions_g = Dense(2, activation='softmax', kernel_regularizer=l2(weight_decay), name='output_gender')(flatten)
    predictions_a = Dense(101, activation='softmax', kernel_regularizer=l2(weight_decay), name='output_age')(flatten)
    predictions_e = Dense(7, activation='softmax', kernel_regularizer=l2(weight_decay), name='output_emotion')(flatten)

    model = Model(inputs=keras_model.layers[0].input, outputs=[predictions_g, predictions_a, predictions_e])
    # model.summary()
    # Total params: 2,821,294
    # Trainable params: 2,787,182
    # Non-trainable params: 34,112

    return model


def get_shufflenet_v2(weight_decay=0.0005):
    kecv_model = kecv_get_model("shufflenetv2_w1", pretrained=False)

    kecv_model.layers.pop()
    for layer in kecv_model.layers:
        layer.trainable = True
    net_layer_len = len(kecv_model.layers)
    flatten = kecv_model.layers[net_layer_len - 1].output
    predictions_g = Dense(2, activation='softmax', kernel_regularizer=l2(weight_decay), name='output_gender')(
        flatten)
    predictions_a = Dense(101, activation='softmax', kernel_regularizer=l2(weight_decay), name='output_age')(
        flatten)
    predictions_e = Dense(7, activation='softmax', kernel_regularizer=l2(weight_decay), name='output_emotion')(
        flatten)
    return Model(inputs=kecv_model.layers[0].input, outputs=[predictions_g, predictions_a, predictions_e])


def get_opconty_shufflenet_v2(weight_decay=0.0005):
    opconty_model = ShuffleNetV2(include_top=False, pooling='avg', input_shape=(64, 64, 3))
    # opconty_model.summary()
    # Total params: 4,018,740
    # Trainable params: 3,990,620
    # Non-trainable params: 28,120

    for layer in opconty_model.layers:
        layer.trainable = True
    net_layer_len = len(opconty_model.layers)
    flatten = opconty_model.layers[net_layer_len - 1].output
    predictions_g = Dense(2, activation='softmax', kernel_regularizer=l2(weight_decay), name='output_gender')(
        flatten)
    predictions_a = Dense(101, activation='softmax', kernel_regularizer=l2(weight_decay), name='output_age')(
        flatten)
    predictions_e = Dense(7, activation='softmax', kernel_regularizer=l2(weight_decay), name='output_emotion')(
        flatten)
    model = Model(inputs=opconty_model.layers[0].input, outputs=[predictions_g, predictions_a, predictions_e])
    # model.summary()
    # Total params: 4,131,490
    # Trainable params: 4,103,370
    # Non-trainable params: 28,120

    return model