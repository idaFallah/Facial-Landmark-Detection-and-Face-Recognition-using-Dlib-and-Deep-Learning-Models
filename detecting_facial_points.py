
# import libs
import dlib
from PIL import Image
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import drive
drive.mount('/content/drive')

face_detector = dlib.get_frontal_face_detector()
points_detector = dlib.shape_predictor('/content/drive/MyDrive/Computer Vision/Weights/shape_predictor_68_face_landmarks.dat')

image = cv2.imread('/content/drive/MyDrive/Computer Vision/Images/people2.jpg')

face_detections = face_detector(image, 1)  # 1 is scale of the image
for face in face_detections:
  points =  points_detector(image, face)
  for points in points.parts():
    cv2.circle(image, (points.x, points.y), 2, (0, 255, 0), 1)

  l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
  cv2.rectangle(image, (l, t), (r, b), (0, 255, 255), 2)

cv2_imshow(image)

# detecting facial descriptors

import os

'''import bz2

# Path to the compressed file
compressed_file = '/content/drive/MyDrive/dlib i downloaded/dlib_face_recognition_resnet_model_v1.dat.bz2'
# Path where you want to save the decompressed file
decompressed_file = '/content/drive/MyDrive/dlib i downloaded/dlib_face_recognition_resnet_model_v1.dat'

# Decompress the file
with bz2.open(compressed_file, 'rb') as f_in:
    with open(decompressed_file, 'wb') as f_out:
        f_out.write(f_in.read())
'''

#! pip uninstall dlib
#! pip install dlib --no-binary :all:

face_detector = dlib.get_frontal_face_detector()
points_detector = dlib.shape_predictor('/content/drive/MyDrive/Computer Vision/Weights/shape_predictor_68_face_landmarks.dat')
face_descriptor_extractor = dlib.face_recognition_model_v1('/content/drive/MyDrive/Computer Vision/Weights/dlib_face_recognition_resnet_model_v1.dat')

import zipfile
path = '/content/drive/MyDrive/Computer Vision/Datasets/yalefaces.zip'
zip_object = zipfile.ZipFile(file=path, mode='r')
zip_object.extractall('./')
zip_object.close()

index = {}
idx = 0
face_descriptors = None

paths = [os.path.join('/content/yalefaces/train', f) for f in os.listdir('/content/yalefaces/train')]
for path in paths:
  #print(path)
  image = Image.open(path).convert('RGB')
  image_np = np.array(image, 'uint8')
  face_detections = face_detector(image_np, 1)
  for face in face_detections:
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(image_np, (l, t), (r, b), (0, 255, 255), 2)

    points = points_detector(image_np, face)
    for part in points.parts():
      cv2.circle(image_np, (part.x, part.y), 2, (0, 255, 0), 1)

    face_descriptor = face_descriptor_extractor.compute_face_descriptor(image_np, points)
    #print(type(face_descriptor))
    #print(len(face_descriptor))
    face_descriptor = [f for f in face_descriptor]
    #print(face_descriptor)
    face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
    #print(face_descriptor)
    #print(face_descriptor.shape)
    face_descriptor = face_descriptor[np.newaxis, :]  # to add new info
    #print(face_descriptor.shape)

    if face_descriptors is None:
      face_descriptors = face_descriptor
    else:
      face_descriptors= np.concatenate((face_descriptors, face_descriptor), axis=0)

    index[idx] = path
    idx += 1
  #cv2_imshow(image_np)

face_descriptors.shape

index

len(index)

#calculating the distance between faces: the lower the distance, the higher the similarity

np.linalg.norm(face_descriptors[131] - face_descriptors[131])

np.linalg.norm(face_descriptors[0] - face_descriptors, axis = 1)

np.argmin(np.linalg.norm(face_descriptors[0] - face_descriptors[1:], axis=1)) # 1: -> to find min distance value other than the image itself

np.linalg.norm(face_descriptors[0] - face_descriptors[1:], axis=1)[84]

# detecting faces with Dlib

threshold = 0.5
predictions = []
expected_outputs = []

paths = [os.path.join('/content/yalefaces/test', f) for f in os.listdir('/content/yalefaces/test')]
for path in paths:
  image = Image.open(path).convert('RGB')
  image_np = np.array(image, 'uint8')
  face_detections = face_detector(image_np, 1)

  for face in face_detections:
    points = points_detector(image_np, face)
    face_descriptor = face_descriptor_extractor.compute_face_descriptor(image_np, points)
    face_descriptor = [f for f in face_descriptor]
    face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
    face_descriptor = face_descriptor[np.newaxis, :]

    distances = np.linalg.norm(face_descriptor - face_descriptors, axis=1)
    min_index = np.argmin(distances)
    min_distance = distances[min_index]

    if min_distance <= threshold:
     pred_name =  int(os.path.split(index[min_index])[1].split('.')[0].replace('subject', ''))
    else:
      pred_name = 'not identified'

    real_name = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))

    predictions.append(pred_name)
    expected_outputs.append(real_name)


    cv2.putText(image_np, 'Pred: '+ str(pred_name), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
    cv2.putText(image_np, 'Exp: '+ str(real_name), (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

cv2_imshow(image_np)
predictions = np.array(predictions)
expected_outputs = np.array(expected_outputs)

predictions

expected_outputs

from sklearn.metrics import accuracy_score
accuracy_score(expected_outputs, predictions)





