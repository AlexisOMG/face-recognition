import os
import sys
import pickle
from typing import Dict, List

from numpy import ndarray
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

    def get_face_encodings(self, people: PeopleImagePaths) -> FacesEncodings:
        res = {}

        for name in people:
            images = tl.load_images(people[name])
            person_encds = []
            for image in images:
                faces = tl.extract_faces(image)
                for face in faces:
                    face_encds = tl.get_face_encoding(face)
                    for encd in face_encds:
                        person_encds.append(encd)

            res[name] = person_encds

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
