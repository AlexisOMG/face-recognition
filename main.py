import sys
import face_recognition as fr
import tools.tools as tl
from mark_data.mark_data import FaceData
import cv2
from network.network import build_network
import tensorflow as tf
from datetime import datetime
import os
from network.network import triplet_loss
from network.datagen import DataGenerator
import numpy as np
from network.datagen import preprocess_input
from tensorflow.keras.applications import vgg16
from tensorflow.keras import Model
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
import time


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

def main():
    with tf.device('CPU'):
        vgg_model = vgg16.VGG16(weights='imagenet')
        netw = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
        # netw = build_network()
        checkpoint = tf.train.Checkpoint(netw)
        checkpoint.restore('./logs/model/siamese-1').expect_partial()
        md = FaceData()
        start = time.time_ns()
        encds = md.get_face_encodings(md.read_dataset(), netw=netw)
        print('Process encds: ', time.time_ns() - start)
        # print(encds)
        # sys.exit(0)
        # md.save_face_encodings_to_cache(encds)
        # encds = md.load_face_encodings_from_cache()
        md.set_faces_encodings(encds)
        # video = cv2.VideoCapture('videoplayback.mp4')
        video = cv2.VideoCapture('videoplayback.mp4')

        id = 0
        while True:
            ret, image = video.read()
            if ret:
                locations = tl.get_face_locations([image])
                for (t, r, b, l) in locations[0]:
                    frame = image[t:b, l:r]
                    frame = cv2.resize(frame, (224, 224))
                    frame = np.asarray(frame, dtype=np.float64)
                    frame = np.expand_dims(frame, axis=0)
                    frame = preprocess_input(frame)
                    start = time.time_ns()
                    faces = netw.predict(frame)
                    name = 'Unknown'
                    min_dist = 1.0
                    min_name = ''
                    for face in faces:
                        face = face / np.linalg.norm(face)

                        if name != 'Unknown':
                            break
                        for known_person in encds:

                            known_face = encds[known_person].mean(axis=0)
                            dist = tl.euclidean_dist(known_face, face)

                            if dist <= min_dist:
                                min_dist = dist
                                min_name = known_person

                    if min_dist < 0.75:
                        name = min_name
                    
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
                print(id, time.time_ns() - start)
                id += 1
                cv2.imshow("fr", image)
                k = cv2.waitKey(20)
                if k == ord("q"):
                        print("Q pressed, closing the app")
                        break
            else:
                print("[Error] Can't get the frame...")
                break


if __name__ == '__main__':
    main()
