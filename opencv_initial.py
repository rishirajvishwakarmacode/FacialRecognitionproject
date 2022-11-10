from re import T
import cv2
import face_recognition
import time

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("D:\\SIH2022\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
rishi_img = face_recognition.load_image_file('known/dp1.jpg')
rishi_img_encodings = face_recognition.face_encodings(rishi_img)[0]
print(rishi_img_encodings)
capture_frame = True

known_face_encodings = [rishi_img_encodings]

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    cv2.imshow('small_frame', rgb_small_frame)

    if capture_frame:
        print('entered loop')
        face_locations = face_recognition.face_locations(rgb_small_frame)
        print('got face locations')
        print(face_locations)
        stream_face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        print(stream_face_encodings)

        for face_encoding in stream_face_encodings:
            match = face_recognition.compare_faces(known_face_encodings, face_encoding)
            print(match)

        capture_frame = not capture_frame

    for (top, right, bottom, left) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "Rishi Raj Vishwakarma", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cap.destroyAllWindows()