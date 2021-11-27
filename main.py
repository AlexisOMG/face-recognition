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
    md = FaceData()
    md.save_face_encodings_to_cache(md.get_face_encodings(md.read_dataset()))
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
    v1()


if __name__ == '__main__':
    main()
