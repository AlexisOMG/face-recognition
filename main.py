from typing import Any, List, Tuple
from PIL import Image, ImageDraw
import face_recognition as fr
from numpy import ndarray
import tools.tools as tl
import mark_data.mark_data as md

def main():
    tl.build_dataset_from_video('video/alexis.mp4', 'alexis')

if __name__ == '__main__':
    main()
