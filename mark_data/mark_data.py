import os
import sys
import pickle
from typing import Dict, List
import cv2
import dlib

import numpy as np
import tensorflow as tf
from network.datagen import preprocess_input

from numpy import ndarray
from network.network import Network
import tools.tools as tl

DEFAULT_PATH_TO_DATASET = 'dataset/'
DEFAULT_PATH_TO_CACHE = '.dataset.pickle'

PeopleImagePaths = Dict[str, List[str]]
FacesEncodings = Dict[str, List[ndarray]]

class FaceData:
    def __init__(self, path_to_dataset: str = DEFAULT_PATH_TO_DATASET, path_to_faces_cache: str = DEFAULT_PATH_TO_CACHE) -> None:
        self.path_to_dataset = path_to_dataset
        if self.path_to_dataset[-1] == '/':
            self.path_to_dataset = self.path_to_dataset[:-1]
        self.path_to_faces_cache = path_to_faces_cache
        self.faces_encodings = {}

    def get_path_to_dataset(self) -> str:
        return self.path_to_dataset

    def set_path_to_dataset(self, path_to_dataset: str) -> None:
        self.path_to_dataset = path_to_dataset

    def get_path_to_faces_cache(self) -> str:
        return self.get_path_to_faces_cache

    def set_path_to_faces_cache(self, path_to_faces_cache: str) -> None:
        self.path_to_faces_cache = path_to_faces_cache

    def get_faces_encodings(self) -> FacesEncodings:
        return self.faces_encodings

    def set_faces_encodings(self, faces_encodings: FacesEncodings) -> None:
        self.faces_encodings = faces_encodings

    def read_dataset(self) -> PeopleImagePaths:
        if not os.path.exists(self.path_to_dataset):
            print("Error: No dataset directory")
            sys.exit(-1)

        people_image_paths = {}

        for name in os.listdir(self.path_to_dataset):
            if not os.path.isdir(self.path_to_dataset + '/' + name):
                continue
            if len(people_image_paths) == 0 or name not in people_image_paths:
                people_image_paths[name] = []
            
            path_to_person_images = self.path_to_dataset + '/' + name
            for path in os.listdir(path_to_person_images):
                people_image_paths[name].append(path_to_person_images + '/' + path)

        return people_image_paths

    def get_face_encodings(self, people: PeopleImagePaths, netw: Network) -> FacesEncodings:
        res = {}
        face_detector = dlib.get_frontal_face_detector()

        for name in people:
            person_face_encodings = []
            images = []
            for image_path in people[name]:
                img = cv2.imread(image_path)
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detector(img, 0)
                for face in faces:
                    t, r, b, l = max(0, face.top()), min(img.shape[1], face.right()), min(img.shape[0], face.bottom()), max(0, face.left())
                    if t >= 0 and r >= 0 and b >= 0 and l >= 0:
                        frame = img[t:b, l:r]
                        frame = cv2.resize(frame, (224, 224))
                        frame = np.asarray(frame, dtype=np.float64)
                        frame = np.expand_dims(frame, axis=0)
                        frame = preprocess_input(frame)
                        features = netw.predict(frame)
                        # print(features)
                        for feature in features:
                            feature = feature / np.linalg.norm(feature)
                            person_face_encodings.append(feature)

            res[name] = np.asarray(person_face_encodings, dtype=np.float64)

        return res

    def save_face_encodings_to_cache(self, encds: FacesEncodings) -> None:
        with open(self.path_to_faces_cache, 'wb') as f:
            pickle.dump(encds, f)
    
    def load_face_encodings_from_cache(self) -> FacesEncodings:
        return pickle.loads(open(self.path_to_faces_cache, 'rb').read())

    def recognize_face(self, face: ndarray) -> str:
        for name in self.faces_encodings:
            if tl.compare_faces(self.faces_encodings[name], face):
                return name
        
        return 'Unknown'
