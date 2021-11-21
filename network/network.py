import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, layers, backend
from tensorflow.keras.applications import vgg16
from network.datagen import DataGenerator
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
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
    vgg = load_model('logs/premodel/siamese-1')
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
    # print(t.summary())

    
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

def train_network():
    # device_name = tf.test.gpu_device_name()
    # if device_name != '/device:GPU:0':
    #     raise SystemError('GPU device not found')
    # print('Found GPU at: {}'.format(device_name))
    # for gpu in tf.config.list_physical_devices('GPU'):
    #     tf.config.experimental.set_memory_growth(gpu, True)
    # preproccess_data()
    # clear_data()
    # sys.exit(-1)
    # tf.debugging.set_log_device_placement(True)
    epochs = 10

    optimizer = Adam()
    binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()

    base_dir = "."

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(base_dir, 'logs/func/%s' % stamp)
    writer = tf.summary.create_file_writer(logdir)

    scalar_logdir = os.path.join(base_dir, 'logs/scalars/%s' % stamp)
    file_writer = tf.summary.create_file_writer(scalar_logdir + "/metrics")

    checkpoint_path = os.path.join(base_dir, 'logs/model/siamese')

    netw = build_network()

    def train(X):
        with tf.GradientTape() as tape:
            y_pred = netw(X)
            loss = triplet_loss(y_pred)
        grad = tape.gradient(loss, netw.trainable_variables)
        optimizer.apply_gradients(zip(grad, netw.trainable_variables))
        return loss

    data_generator = DataGenerator(dataset_path='./vgg_face_dataset/')
    a, p, n = data_generator[0]
    checkpoint = tf.train.Checkpoint(model=netw)
    losses = []
    accuracy = []

    no_of_batches = data_generator.__len__()
    print(no_of_batches)

    # try:
    #     with tf.device('/device:GPU:0'):
    for i in range(1, epochs+1, 1):
        loss = 0
        with tqdm(total=no_of_batches) as pbar:
            
            description = "Epoch " + str(i) + "/" + str(epochs)
            pbar.set_description_str(description)
            
            for j in range(no_of_batches):
                data = data_generator[j]
                temp = train(data)
                loss += temp
                
                pbar.update()
                print_statement = "Loss :" + str(temp.numpy())
                pbar.set_postfix_str(print_statement)
            
            loss /= no_of_batches
            losses.append(loss.numpy())
            with file_writer.as_default():
                tf.summary.scalar('Loss', data=loss.numpy(), step=i)
                
            print_statement = "Loss :" + str(loss.numpy())
            
            pbar.set_postfix_str(print_statement)
    # except RuntimeError as e:
    #     print(e)

    checkpoint.save(checkpoint_path)
    print("Checkpoint Saved")