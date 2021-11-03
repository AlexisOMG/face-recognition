import os
import sys
import json
import pickle

import tools.tools as tl

PATH_TO_DATASET = 'dataset'

def load_dataset():
    if not os.path.exists(PATH_TO_DATASET):
        print("Error: No dataset directory")
        sys.exit(-1)

    image_paths = [PATH_TO_DATASET + '/' + path for path in os.listdir(PATH_TO_DATASET)]
    images = tl.load_images(image_paths)

    data = {}
    ind = 0

    for (i, image) in enumerate(images):
        encd = tl.get_face_encoding(image)

        if len(data) == 0:
            data[ind] = {
                'encodings': [encd], 
                'file_paths': [image_paths[i]],
            }
            ind += 1
        else:
            added = False
            for key in data:
                print(key)
                for face_encd in data[key]['encodings']:
                    if tl.compare_faces(face_encd, encd)[0]:
                        data[key]['encodings'].append(encd)
                        data[key]['file_paths'].append(image_paths[i])
                        added = True
            if not added:
                data[ind] = {
                    'encodings': [encd], 
                    'file_paths': [image_paths[i]],
                }
                ind += 1

    with open('.dataset.pickle', 'wb') as f:
        pickle.dump(data, f)
    return data

