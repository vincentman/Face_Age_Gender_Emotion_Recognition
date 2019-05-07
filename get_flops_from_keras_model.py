import keras.backend as K
import tensorflow as tf
import keras
from train_shufflenet import myMAE


model = keras.models.load_model('my_trained/18/model.h5', custom_objects={'myMAE': myMAE})
sess = K.get_session()
graph = sess.graph


with graph.as_default():
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPS: {:,}'.format(flops.total_float_ops))
    # FLOPS: 38,069,010 for my_trained/18/model.h5


