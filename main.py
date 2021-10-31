from typing import Any, List, Tuple
from PIL import Image, ImageDraw
import face_recognition as fr
from numpy import ndarray

def load_images(paths: List[str]) -> List[ndarray]:
    return [fr.load_image_file(path) for path in paths]

def get_face_locations(images: List[ndarray]) -> List[List[Tuple[int, Any, Any, int]]]:
    return [fr.face_locations(image) for image in images]

def highlight_faces(images: List[ndarray]) -> None:
    face_locations = get_face_locations(images)
    for i in range(len(face_locations)):
        img = Image.fromarray(images[i])
        draw = ImageDraw.Draw(img)
        for (t, r, b, l) in face_locations[i]:
            draw.rectangle(((l, t), (r, b)), outline=(255, 255, 0), width=4)
        del draw
        img.save(f'imgs/highlighted_faces_{i}.jpg')

def extract_faces(images: List[ndarray]) -> None:
    face_locations = get_face_locations(images)
    cnt = 0
    for i in range(len(face_locations)):
        for (t, r, b, l) in face_locations[i]:
            face = images[i][t:b, l:r]
            face_img = Image.fromarray(face)
            face_img.save(f'imgs/face_{cnt}.jpg')
            cnt += 1

def compare_faces(known_face: ndarray, unknown_face: ndarray) -> List[bool]:
    known_face_encd = fr.face_encodings(known_face)[0]
    unknown_face_encd = fr.face_encodings(unknown_face)[0]
    return fr.compare_faces([known_face_encd], unknown_face_encd)

def main():
    images = load_images(['imgs/img.jpg'])
    highlight_faces(images)
    extract_faces(images)
    print(compare_faces(images[0], images[0]))

if __name__ == '__main__':
    main()
