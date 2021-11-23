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
        netw = build_network()
        checkpoint = tf.train.Checkpoint(netw)
        checkpoint.restore('./logs/model/siamese-1').expect_partial()
        md = FaceData()
        encds = md.get_face_encodings(md.read_dataset(), netw=netw)
        print(encds)
        md.save_face_encodings_to_cache(encds)
        encds = md.load_face_encodings_from_cache()
        md.set_faces_encodings(encds)
        video = cv2.VideoCapture(0)

        while True:
            ret, image = video.read()
            if ret:
                locations = fr.face_locations(image)
                for (t, r, b, l) in locations:
                    frame = image[t:b, l:r]
                    frame = cv2.resize(frame, (224, 224))
                    frame = np.asarray(frame, dtype=np.float64)
                    frame = np.expand_dims(frame, axis=0)
                    frame = preprocess_input(frame)
                    faces = netw.get_features(frame)
                    faces = tf.math.l2_normalize(faces, axis=-1)
                    # print(face)
                    name = 'whoisit'
                    for face in faces:
                        for alex in encds['Alexey']:
                            dist = tf.norm(alex-face, ord='euclidean')
                            print(dist)
                            # loc = tf.argmin(dist)
                            # print(dist[loc])
                            if dist <= 0.7:
                                name = 'alexis'
                                break
                        if name == 'alexis':
                            break

                    # face = np.asarray(face[0], dtype=np.float64)
                    # # face = np.expand_dims(face, axis=0)
                    # # print(face.dtype)
                    # name = md.recognize_face(face=face)
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
                k = cv2.waitKey(10)
                if k == ord("q"):
                        print("Q pressed, closing the app")
                        break
            else:
                print("[Error] Can't get the frame...")
                break


if __name__ == '__main__':
    main()
