from typing import Any, List, Tuple
from PIL import Image, ImageDraw
import face_recognition as fr
from numpy import ndarray
import tools.tools as tl
from mark_data.mark_data import load_dataset

def main():
    # images = tl.load_images(['imgs/img.jpg'])
    # tl.highlight_faces(images)
    # tl.extract_faces(images)
    # print(tl.compare_faces(tl.get_face_encoding(images[0]), tl.get_face_encoding(images[0])))
    print(load_dataset())

if __name__ == '__main__':
    main()
