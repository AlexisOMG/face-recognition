import os
import sys
import pickle
from typing import Dict, List

from numpy import ndarray

import tools.tools as tl

PATH_TO_DATASET = 'dataset/'
PATH_TO_BIN = '.dataset.pickle'

def load_dataset() -> Dict[str, List[str]]:
    if not os.path.exists(PATH_TO_DATASET):
        print("Error: No dataset directory")
        sys.exit(-1)

    paths_to_people = {}

    for name in os.listdir(PATH_TO_DATASET):
        if len(paths_to_people) == 0 or name not in paths_to_people:
            paths_to_people[name] = []

        paths_to_people[name] += [PATH_TO_DATASET + name + '/' + path for path in os.listdir(PATH_TO_DATASET + name)]
    
    return paths_to_people

def get_face_encds(people: Dict[str, List[str]]) -> Dict[str, List[ndarray]]:
    res = {}

    for name in people:
        person_images = tl.load_images(people[name])
        person_face_encodings = [tl.get_face_encoding(face) for face in tl.extract_faces(person_images)]
        res[name] = person_face_encodings

    return res

def save_face_encds(encds: Dict[str, List[ndarray]]) -> None:
    with open(PATH_TO_BIN, 'wb') as f:
        pickle.dump(encds, f)

def load_face_encds() -> Dict[str, List[ndarray]]:
    return pickle.loads(open(PATH_TO_BIN, 'rb').read())

def match_face(known_faces: Dict[str, List[ndarray]], face: ndarray) -> str:
    for name in known_faces:
        if tl.compare_faces(known_faces[name], face):
            return name
    return "Unknown"
