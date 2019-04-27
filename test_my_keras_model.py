from my_keras_model import get_mobilenet_v2, get_shufflenet_v2, get_opconty_shufflenet_v2
import numpy as np

# model = get_mobilenet_v2()
# print('my mobilenet model: {:,} params'.format(model.count_params()))
# x = np.zeros((1, 64, 64, 3), np.float32)
# y = model.predict(x)
# print('my mobilenet model: prediction finished')

# model = get_shufflenet_v2()
# print('my shufflenet model: {:,} params'.format(model.count_params()))
# x = np.zeros((1, 224, 224, 3), np.float32)
# y = model.predict(x)
# print('my mobilenet model: prediction finished')

model = get_opconty_shufflenet_v2()
print('opconty shuffle_model: {:,} params'.format(model.count_params()))
x = np.zeros((1, 64, 64, 3), np.float32)
y = model.predict(x)
print('opconty shuffle_model: prediction finished')






