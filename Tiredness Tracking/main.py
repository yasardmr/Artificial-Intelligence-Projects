import cv2
import dlib
from datetime import datetime
from time import time, sleep
from scipy.spatial import distance
from logging import *
import numpy as np

text_log(message='Imported libraries successfully.',
         show_console=True)

# Global Constants

DEBUG_FACE = False
DEBUG_LANDMARKS = True
DEBUG_BLINK = True
CLEAR_OLD_LOGS = False

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
HAAR_DATA = cv2.CascadeClassifier('data/frontfacedata.xml')
FONT = cv2.FONT_HERSHEY_SIMPLEX

try:
    text_log(message='Opening camera')
    VID = cv2.VideoCapture(0)

except KeyboardInterrupt:
    exit('Quitting')

except Exception as e:

    text_log(message='Error while reading camera, quitting',
             show_console=True)
    exit('Quitting')

if CLEAR_OLD_LOGS:
    clear_logs()

text_log(message='Global variables set.')


def find_face() -> list:

    try:
        ret, frame = VID.read()

        if ret:
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_coordinates = HAAR_DATA.detectMultiScale(grayscale)

            if len(face_coordinates) == 1:
                x, y, w, h = [each for each in face_coordinates[0]]
                # return [x, y, x + w, y + h]  # Face Mask

                subframe = grayscale[y: y + h, x: x + w]

            else:
                x, y = 0, 0
                w = grayscale.shape[1]
                h = grayscale.shape[0]
                subframe = grayscale


            if DEBUG_FACE:

                try:
                    print(len(face_coordinates), face_coordinates)
                    # print(f'X {x} Y {y}, W {w}, H {h}')
                    cv2.imshow('Raw Input', frame)
                    cv2.rectangle(img=grayscale,
                                  pt1=(x, y),
                                  pt2=(x+h, y+w),
                                  thickness=2,
                                  color=(255, 255, 255))
                    cv2.imshow('Grayscale', grayscale)
                    cv2.imshow('Extracted Face', subframe)
                    cv2.waitKey(1)
                    # print(face_coordinates)

                except KeyboardInterrupt:
                    exit_sequence()

                except Exception as e:
                    text_log(message=f'Failed to debug face - {e}',
                             show_console=True)

            return [frame, subframe, [x, y, w, h]]

    except KeyboardInterrupt:
        exit_sequence()

    except Exception as e:
        text_log(message=f'Could not find video - {e}',
                 show_console=True)


def calculate_EAR(eye) -> float:

    try:
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear_aspect_ratio = (A + B) / (2.0 * C)
        return ear_aspect_ratio

    except KeyboardInterrupt:
        exit_sequence()

    except Exception as e:
        text_log(message=f'Failed to calculate EAR {e}')


def check_blink(grey_frame) -> bool:

    try:
        blink = False
        faces = DETECTOR(grey_frame)

        for face in faces:

            face_landmarks = PREDICTOR(grey_frame, face)
            leftEye = []
            rightEye = []

            for n in range(36, 42):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                leftEye.append((x, y))
                next_point = n + 1
                if n == 41:
                    next_point = 36
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y

                if DEBUG_BLINK:
                    cv2.line(grey_frame, (x, y), (x2, y2), (150, 150, 0), 2)

            for n in range(42, 48):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                rightEye.append((x, y))
                next_point = n + 1
                if n == 47:
                    next_point = 42
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y

                if DEBUG_BLINK:
                    cv2.line(grey_frame, (x, y), (x2, y2), (150, 150, 0), 2)

            try:
                if DEBUG_BLINK:
                    cv2.imshow('Eyes', grey_frame)
                    cv2.waitKey(1)

                left_ear = calculate_EAR(leftEye)
                right_ear = calculate_EAR(rightEye)

                EAR = (left_ear + right_ear) / 2
                EAR = round(EAR, 2)

                if EAR < 0.18:
                    blink = True
                    text_log(message='Blink Detected')
                    sleep(0.05)

            except KeyboardInterrupt:
                exit_sequence()

            except Exception as e:
                text_log(message=f'Failed to debug blink {e}',
                         show_console=True)

            return blink

    except KeyboardInterrupt:
        exit_sequence()

    except Exception as e:
        text_log(message=f'Failed to check blinks {e}',
                 show_console=True)


def check_yawn():
    pass


def calibrate(duration: float) -> list:
    ret, frame = VID.read()

    if ret:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = DETECTOR(grey)
        # frame = cv2.resize(frame, dsize=(frame.shape[1] * 2, frame.shape[0] * 2))

        lt_eye = []
        rt_eye = []

        start = time()

        while True:

            for face in faces:
                landmarks = PREDICTOR(grey, face)

                points = []

                for n in range(0, 68):

                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    # Appending Y coordinates only since we need vertical euclidean distance between eyelids/lips
                    points.append(y)
                    # print(points)

                    if ((n > 35) and (n < 48)) or ((n > 47) and (n < 68)):

                        if DEBUG_LANDMARKS:
                            cv2.circle(frame,
                                       center=(x, y),
                                       radius=2,
                                       color=(0, 0, 0),
                                       thickness=-1)

                            add_text(winname=frame,
                                     location=(x, y),
                                     message=str(n),
                                     colour=(255, 255, 255),
                                     size=0.25
                                     )

                add_text(winname=frame,
                         message='Calibrating system, please relax your face to a normal position',
                         thick=2,
                         size=0.5,
                         colour=(25, 25, 25)
                         )

                lt_avg = ((points[41] - points[37]) + (points[40] - points[38]))/2
                rt_avg = ((points[47] - points[43]) + (points[48] - points[44]))/2

                lt_eye.append(lt_avg)
                rt_eye.append(rt_avg)

                cv2.imshow("Facial Landmarks", frame)
                cv2.waitKey(1)

                if time()-start <= duration:
                    break

                break

                # print(lt_eye, rt_eye)

        cv2.destroyAllWindows()

        # Return average eyelid separation for both eyes
        return [sum(lt_eye)/len(lt_eye), sum(rt_eye)/len(rt_eye)]


def exit_sequence() -> None:
    cv2.destroyAllWindows()
    del DETECTOR, PREDICTOR, HAAR_DATA, DEBUG_FACE, DEBUG_LANDMARKS, DEBUG_BLINK, CLEAR_OLD_LOGS

    text_log(message='Quit',
             curr_time=datetime.now().strftime("%H:%M:%S"))

    quit('Quitting')


def add_text(winname, message, location=(35, 35), colour=(255, 255, 255), thick=1, size=1.0) -> None:

    try:
        cv2.putText(img=winname,
                    text=message,
                    org=location,
                    color=colour,
                    fontScale=size,
                    fontFace=FONT,
                    thickness=thick)

    except KeyboardInterrupt:
        exit_sequence()

    except Exception as e:
        text_log(f'Failed to add text {message} on {winname}. Exception - {e}',
                 curr_time=datetime.now().strftime("%H:%M:%S"),
                 show_console=True)


text_log(message='All functions initialized, starting...',
         show_console=True)

blinks = 0
new_blinks = 0
drowsy = False
init = time()

while True:

    face_data = find_face()

    if check_blink(face_data[1]) == True:
        blinks += 1
        new_blinks += 1

    if time() - init < 5 and new_blinks > 10:
        drowsy = True
        new_blinks = 0
        init = time()

    if time() - init >= 5:
        init = time()

    frame = face_data[0]
    x, y, w, h = (i for i in face_data[2])

    add_text(winname=frame,
             message=f'Blinks {blinks}',
             location=(20, 25),
             size=0.75,
             thick=2,
             colour=(0, 0, 0))

    if drowsy:

        text_log(message='Drowsy')
        add_text(winname=frame,
                 message=f'Drowsy {drowsy}',
                 location=(20, 50),
                 size=0.75,
                 thick=2,
                 colour=(0, 0, 200))

    else:

        add_text(winname=frame,
                 message=f'Drowsy {drowsy}',
                 location=(20, 50),
                 size=0.75,
                 thick=2,
                 colour=(0, 255, 0))

    cv2.rectangle(img=frame,
                  pt1=(x, y),
                  pt2=(x + h, y + w),
                  thickness=2,
                  color=(255, 255, 255))

    cv2.imshow('Output', frame)
    cv2.waitKey(1)
