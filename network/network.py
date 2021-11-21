import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras import Model, layers, backend
from keras.applications import vgg16
import os

class Network(Model):
    def __init__(self, vgg: Sequential) -> None:
        super(Network, self).__init__()
        self.vgg = vgg

    @tf.function
    def call(self, inputs):
        face1, face2, face3 = inputs
        with tf.name_scope('Anchor') as scope:
            feature1 = self.vgg(face1)
            feature1 = tf.math.l2_normalize(feature1, axis=-1)
        with tf.name_scope('Positive') as scope:
            feature2 = self.vgg(face2)
            feature2 = tf.math.l2_normalize(feature2, axis=-1)
        with tf.name_scope('Negative') as scope:
            feature3 = self.vgg(face3)
            feature3 = tf.math.l2_normalize(feature3, axis=-1)
        return [feature1, feature2, feature3]

    @tf.function
    def get_features(self, inputs):
        return tf.math.l2_normalize(self.vgg(inputs), axis=-1)

def build_network() -> Network:
    # vgg_model = vgg16.VGG16()
    # vgg = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
    # vgg.save('logs/model/siamese-1', save_format='tf')
    # vgg.save_weights('vgg_face_weights.h5')
    vgg = load_model('logs/model/siamese-1')
    t = Sequential(vgg.layers)
    # vgg = Sequential()
    # vgg.add(layers.Convolution2D(64, (3, 3), activation='relu', padding="SAME", input_shape=(224,224, 3)))
    # vgg.add(layers.Convolution2D(64, (3, 3), activation='relu', padding="SAME"))
    # vgg.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    
    # vgg.add(layers.Convolution2D(128, (3, 3), activation='relu', padding="SAME"))
    # vgg.add(layers.Convolution2D(128, (3, 3), activation='relu', padding="SAME"))
    # vgg.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    
    # vgg.add(layers.Convolution2D(256, (3, 3), activation='relu', padding="SAME"))
    # vgg.add(layers.Convolution2D(256, (3, 3), activation='relu', padding="SAME"))
    # vgg.add(layers.Convolution2D(256, (3, 3), activation='relu', padding="SAME"))
    # vgg.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    
    # vgg.add(layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    # vgg.add(layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    # vgg.add(layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    # vgg.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    
    # vgg.add(layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    # vgg.add(layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    # vgg.add(layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    # vgg.add(layers.MaxPooling2D((2,2), strides=(2,2)))

    # t.add(layers.Flatten())

    # vgg.add(layers.Dense(4096, activation='relu'))
    # vgg.add(layers.Dropout(0.5))
    # vgg.add(layers.Dense(4096, activation='relu'))
    # vgg.add(layers.Dropout(0.5))
    # vgg.add(layers.Dense(2622, activation='softmax'))
    base_dir = "."
    t.load_weights(os.path.join(base_dir, 'vgg_face_weights.h5'))


    # vgg.pop()
    t.add(layers.Dense(128, use_bias=False))

    for layer in t.layers[:-1]:
        layer.trainable = False

    network = Network(vgg=t)
    print(t.summary())

    
    # checkpoint_path = os.path.join(base_dir, 'logs/model/siamese-1')

    # _ = network([tf.zeros((1,224,224,3)), tf.zeros((1,224,224,3)), tf.zeros((1,224,224,3))])
    # _ = network.get_features(tf.zeros((1,224,224,3)))

    # checkpoint = tf.train.Checkpoint(model=network)
    # checkpoint.restore(checkpoint_path)

    return network

def triplet_loss(x, alpha = 0.2):
    anchor,positive,negative = x
    pos_dist = backend.sum(backend.square(anchor-positive),axis=1)
    neg_dist = backend.sum(backend.square(anchor-negative),axis=1)
    basic_loss = pos_dist-neg_dist+alpha
    loss = backend.mean(backend.maximum(basic_loss,0.0))
    return loss
