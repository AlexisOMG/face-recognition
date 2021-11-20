from typing import Any, List, Tuple
from PIL import Image, ImageDraw
import face_recognition as fr
from numpy import ndarray
import tools.tools as tl
from mark_data.mark_data import FaceData
import cv2
from network.network import build_network

def main():
    # tl.build_dataset_from_video('video/alexis.mp4', 'alexis')
    # dt = md.load_dataset()
    # encds = md.get_face_encds(dt)
    # md.save_face_encds(encds)
    # md = FaceData()
    # encds = md.load_face_encodings_from_cache()
    # md.set_faces_encodings(encds)
    # video = cv2.VideoCapture(0)

    # while True:
    #     ret, image = video.read()

    #     if ret:

    #         locations = fr.face_locations(image)
    #         encodings = fr.face_encodings(image, locations)

    #         for face_encoding, face_location in zip(encodings, locations):
    #             name = md.recognize_face(face_encoding)
    #             t, r, b, l = face_location
    #             cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 4)
    #             cv2.rectangle(image, (l, b), (r, b), (0, 255, 0), cv2.FILLED)
    #             cv2.putText(
    #                 image,
    #                 name,
    #                 (l + 10, b + 15),
    #                 cv2.FONT_HERSHEY_TRIPLEX,
    #                 1,
    #                 (255, 255, 255),
    #                 4
    #             )
    #         cv2.imshow("fr", image)
    #         k = cv2.waitKey(20)
    #         if k == ord("q"):
    #                 print("Q pressed, closing the app")
    #                 break
        
    #     else:
    #         print("[Error] Can't get the frame...")
    #         break
    netw = build_network()

if __name__ == '__main__':
    main()
