from typing import Any, List, Tuple
from PIL import Image, ImageDraw
import face_recognition as fr
from numpy import ndarray
import numpy
import tools.tools as tl
from mark_data.mark_data import FaceData
import cv2
from network.network import build_network
import tensorflow as tf
from datetime import datetime
import os
from network.network import triplet_loss
from network.datagen import DataGenerator
from tqdm import tqdm
import dlib
from tensorflow.keras.optimizers import Adam
import shutil


def v1():
    # tl.build_dataset_from_video('video/alexis.mp4', 'alexis')
    # dt = md.load_dataset()
    # encds = md.get_face_encds(dt)
    # md.save_face_encds(encds)
    md = FaceData()
    encds = md.load_face_encodings_from_cache()
    md.set_faces_encodings(encds)
    video = cv2.VideoCapture(0)

    while True:
        ret, image = video.read()

        if ret:

            locations = fr.face_locations(image)
            encodings = fr.face_encodings(image, locations)

            for face_encoding, face_location in zip(encodings, locations):
                name = md.recognize_face(face_encoding)
                t, r, b, l = face_location
                cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 4)
                cv2.rectangle(image, (l, b), (r, b), (0, 255, 0), cv2.FILLED)
                cv2.putText(
                    image,
                    name,
                    (l + 10, b + 15),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1,
                    (255, 255, 255),
                    4
                )
            cv2.imshow("fr", image)
            k = cv2.waitKey(20)
            if k == ord("q"):
                    print("Q pressed, closing the app")
                    break
        
        else:
            print("[Error] Can't get the frame...")
            break

def preproccess_data():
    face_detector = dlib.get_frontal_face_detector()
    dataset_path = './vgg_face_dataset/images'
    path = './vgg_face_dataset/data'
    list_of_images = []
    for dirname in tqdm(os.listdir(path)):
        image_folder_path = os.path.join(path, dirname)
        os.mkdir(os.path.join(dataset_path, dirname))
        for image in tqdm(os.listdir(image_folder_path), leave=True, position=1):
            image_path = os.path.join(image_folder_path, image)
            img = cv2.imread(image_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray, 0)
            if len(faces) == 1:
                for face in faces:
                    t, r, b, l = max(0, face.top()), min(gray.shape[1], face.right()), min(gray.shape[0], face.bottom()), max(0, face.left())
                    if t >= 0 and r >= 0 and b >= 0 and l >= 0:
                        frame = img[t:b, l:r]
                        save_image = os.path.join(os.path.join(dataset_path, dirname), image)
                        cv2.imwrite(save_image, frame)
                        list_of_images.append(dirname + '/' + image)
    with open('./vgg_face_dataset/list.txt', 'w') as f:
        for item in list_of_images:
            f.write("%s\n" % item)

def clear_data():
    list_of_images = []
    dataset_path = './vgg_face_dataset/images'
    for dirname in os.listdir(dataset_path):
        if len(os.listdir(dataset_path+'/'+dirname)) < 3:
            print(f'DELETING {dirname}')
            shutil.rmtree(dataset_path+'/'+dirname)
        else:
            for name in os.listdir(dataset_path+'/'+dirname):
                list_of_images.append(dirname+'/'+name)
    with open('./vgg_face_dataset/list.txt', 'w') as f:
        for item in list_of_images:
            f.write("%s\n" % item)

def main():
    # preproccess_data()
    clear_data()
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

    checkpoint.save(checkpoint_path)
    print("Checkpoint Saved")

if __name__ == '__main__':
    main()
