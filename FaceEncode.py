import cv2
import face_recognition as fr
import numpy as np

def faceencoding(path):
    image = fr.load_image_file(path)
    encoding = fr.face_encodings(image)[0]
    return encoding

if __name__ == "__main__":
    priyaEncoding = faceencoding("Images\Kajal.jpeg")
    print(priyaEncoding)