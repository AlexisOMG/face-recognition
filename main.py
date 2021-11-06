from typing import Any, List, Tuple
from PIL import Image, ImageDraw
import face_recognition as fr
from numpy import ndarray
import tools.tools as tl
import mark_data.mark_data as md

def main():
    ppl = md.load_dataset()
    encds = md.get_face_encds(ppl)
    md.save_face_encds(encds)
    print(md.load_face_encds())

if __name__ == '__main__':
    main()
