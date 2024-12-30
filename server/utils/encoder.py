import os
import tensorflow as tf
import cv2
from tensorflow import keras


class Encoder:
    def __init__(self, model_path:str='model.keras'):
        
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"File: '{model_path}' doesn't exist.")
        try:
            self.model = keras.models.load_model('model.keras')
        except Exception as e:
            print(f"Unable to load file: \n{e}")
    
    def get_faceboxes(self, img):
        '''
        Return list of face bounding box coordinates with eye coordinates

        Args:
            img: numpy.ndrray() assuming input image is rgb format
        Returns:
            faces: numpy.ndrray(n, 4): where n is no of faces presetn in the image  
        '''
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        gray_image = cv2.equalizeHist(gray_image)
        face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        face_boxes = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)
        )
        return face_boxes

    def _resize_imgs(self, img, faceboxes, size=(224, 224)):
        '''
        Returns resized images of size 224 by 224 

        Args:
            boxes ndrray(n, 4): n is number of faces and each face has [x, y, w, h]
            img ndrray(): image with faces
            size tuple(h, w)
        Return:
            imgs tensor(n, 224, 224, 3)
        '''
        faces = []
        for (x,y,w,h) in faceboxes:
            face = img[y:y+h+1, x: x+w+1, :]
            face = tf.image.resize(face, size)
            faces.append(face)
        
        faces = tf.stack(faces)
        return faces

    def preprocess(self, img):
        faceboxes = self.get_faceboxes(img)
        face_imgs = self._resize_imgs(img, faceboxes)
        return face_imgs
    
    def encode(self, img):
        try:
            face_imgs = self.preprocess(img)
            if len(face_imgs) == 0:
                raise ValueError("No face detected.")
            
            res = self.model.predict(face_imgs)
            return res.tolist()
        except ValueError:
            raise
        except Exception as e:
            print('Some Error in encoding image')