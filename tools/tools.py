from typing import Any, List, Tuple
from PIL import Image, ImageDraw
import face_recognition as fr
from numpy import ndarray
import cv2
import os

def load_images(paths: List[str]) -> List[ndarray]:
    return [fr.load_image_file(path) for path in paths]

def get_face_locations(images: List[ndarray]) -> List[List[Tuple[int, Any, Any, int]]]:
    return [fr.face_locations(image) for image in images]

def get_face_encoding(face: ndarray) -> ndarray:
    return fr.face_encodings(face_image=face)[0]

def highlight_faces(images: List[ndarray]) -> None:
    face_locations = get_face_locations(images)
    for i in range(len(face_locations)):
        img = Image.fromarray(images[i])
        draw = ImageDraw.Draw(img)
        for (t, r, b, l) in face_locations[i]:
            draw.rectangle(((l, t), (r, b)), outline=(255, 255, 0), width=4)
        del draw
        img.save(f'imgs/highlighted_faces_{i}.jpg')

def extract_faces(images: List[ndarray]) -> List[ndarray]:
    res = []
    face_locations = get_face_locations(images)
    for i in range(len(face_locations)):
        for (t, r, b, l) in face_locations[i]:
            res.append(images[i][t:b, l:r])
    return res

def save_faces(faces: List[ndarray]) -> None:
    cnt = 0
    for face in faces:
        face_img = Image.fromarray(face)
        face_img.save(f'imgs/face_{cnt}.jpg')
        cnt += 1

def compare_faces(known_face_encds: List[ndarray], unknown_face_encd: ndarray) -> bool:
    res = fr.compare_faces(known_face_encds, unknown_face_encd)
    for r in res:
        if r:
            return True
    return False

def build_dataset_from_video(path: str, name: str) -> None:
    capture = cv2.VideoCapture(path)
    if not os.path.exists("dataset/"+name):
        os.mkdir("dataset/"+name)
    
    cnt = 0
    while True:
        ret, frame = capture.read()
        fps = int(capture.get(cv2.CAP_PROP_FPS)) / 10
        multiplier = fps * 3
        # print(fps, multiplier)

        if ret:
            frame_id = int(round(capture.get(1)))
            # print(frame_id)
            cv2.imshow("frame", frame)
            k = cv2.waitKey(20)

            if frame_id % multiplier == 0:
                cv2.imwrite(f"dataset/{name}/{cnt}.jpg", frame)
                print(f"Take a screenshot {cnt}")
                cnt += 1
            
            if k == ord("q"):
                print("Q pressed, closing the app")
                break
        else:
            print("[Error] Can't get the frame...")
            break

    capture.release()
    cv2.destroyAllWindows()
