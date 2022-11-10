import cv2
import face_recognition as fr
import numpy as np
import urllib
import pymongo
from faceencoding import getdata



if __name__ == '__main__':
    (kwnfcen_list, kwnname_list) = getdata()
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("http://192.168.105.122:81/stream")
    while True:
        ret, frame = cap.read()
        rgb_frame = frame[:,:,::-1]
        strfcloc = fr.face_locations(rgb_frame)
        strfcen = fr.face_encodings(rgb_frame, strfcloc)

        for (top, right, bottom, left), face_encoding in zip(strfcloc,strfcen):
            matches = fr.compare_faces(known_face_encodings=kwnfcen_list,
                                       face_encoding_to_check=face_encoding,
                                       tolerance=0.6)
            name = 'unknown'
            fcdis = fr.face_distance(kwnfcen_list, face_encoding)
            best_match_index = np.argmin(fcdis)
            if matches[best_match_index]:
                name = kwnname_list[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
            cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0,0,255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1)

        cv2.imshow('FcRg', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cap.destroyAllWindows()