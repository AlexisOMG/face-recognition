import tools.tools as tl
from mark_data.mark_data import FaceData
import cv2
import os
import PySimpleGUI as sg
from shutil import copy


def recognize_with_fr(need_update_cache=False):
    md = FaceData()
    if need_update_cache:
        md.save_face_encodings_to_cache(md.get_face_encodings(md.read_dataset()))
    encds = md.load_face_encodings_from_cache()
    md.set_faces_encodings(encds)
    video = cv2.VideoCapture(0)

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
                cv2.rectangle(image, (l, b), (r, b), (0, 255, 0), cv2.FILLED)
                cv2.putText(
                    image,
                    name,
                    (l + 10, b + 15),
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

def ui() -> bool:
    file_list_column = [
        [
            sg.Text('Имя человека'),
            sg.In(size=(25,1), enable_events=True, key='-PERSON NAME-'),
        ],
        [
            sg.Text('Директория с фото'),
            sg.In(size=(24, 1), enable_events=True, key='-FOLDER-'),
            sg.FolderBrowse(),
        ],
        [
            sg.Listbox(
                values=[], enable_events=True, size=(40, 20), key='-FILE LIST-'
            )
        ],
        [
            sg.Button(button_text='Добавить человека', enable_events=True, key='-ADD PERSON-'),
            sg.Button(button_text='Начать распознавание', enable_events=True, key='-RECOGNIZE-'),
        ],
    ]
    
    person_name = ''
    files = []
    need_update_cache = False
    
    window = sg.Window(title='Распознавание', layout=file_list_column, resizable=True)
    while True:
        ev, val = window.read()
    
        if ev == sg.WIN_CLOSED or ev == '-RECOGNIZE-':
            break
    
        if ev == '-FOLDER-':
            folder = val['-FOLDER-']
            try:
                file_list = os.listdir(folder)
            except:
                file_list = []
    
            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith((".png", ".gif", '.jpg', '.jpeg'))
            ]
            if len(fnames) < 10:
                window['-FILE LIST-'].update(['Need at least 10 photos of person'])
            else:
                window['-FILE LIST-'].update(fnames)
                base_dir = folder if folder[-1] == '/' else folder + '/'
                files = [base_dir + f for f in fnames]
        elif ev == '-PERSON NAME-':
            person_name = val['-PERSON NAME-']
        elif ev == '-ADD PERSON-':
            if person_name == None or person_name == '':
                print('ERROR: No name specified')
            if files == None or len(files) == 0:
                print('ERROR: No photos')

            person_dir = 'dataset/'+person_name
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)

            for file in files:
                copy(file, person_dir)
            need_update_cache = True
            person_name = ''
            files = []
            window['-PERSON NAME-'].update('')
            window['-FOLDER-'].update('')
            window['-FILE LIST-'].update([])
    
    window.close()
    return need_update_cache

def main():
    need_update_cache = ui()

    recognize_with_fr(need_update_cache)


if __name__ == '__main__':
    main()
