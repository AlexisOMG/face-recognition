import tools.tools as tl
from face_data.face_data import FaceData
import cv2

def recognize_with_fr(need_update_cache=False, path_to_video = None):
    md = FaceData()
    if need_update_cache:
        md.save_face_encodings_to_cache(md.get_face_encodings(md.read_dataset()))
    encds = md.load_face_encodings_from_cache()
    md.set_faces_encodings(encds)
    video = None
    if path_to_video is None or path_to_video == '':
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(path_to_video)

    while True:
        ret, image = video.read()

        if ret:

            small_image = cv2.resize(image, None, fx=0.5, fy=0.5)
            rgb_image = small_image[:, :, ::-1]
            
            locations = tl.get_face_locations(image=rgb_image)
            encodings = tl.get_face_encoding(face=rgb_image, locations=locations)

            for face_encoding, face_location in zip(encodings, locations):
                name = md.recognize_face(face_encoding)
                t, r, b, l = face_location
                t *= 2
                r *= 2
                b *= 2
                l *= 2
                cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 4)
                cv2.rectangle(image, (l, b), (r, b+20), (0, 255, 0), cv2.FILLED)
                cv2.putText(
                    image,
                    name,
                    (l + 15, b + 15),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1,
                    (255, 255, 255),
                    4
                )
            cv2.imshow('recognition', image)
            k = cv2.waitKey(20)
            if k == ord("q"):
                    print("Q pressed, closing the app")
                    break
        
        else:
            print("[Error] Can't get the frame...")
            break
