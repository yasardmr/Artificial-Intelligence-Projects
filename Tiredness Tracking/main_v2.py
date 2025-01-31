import dlib
import cv2
import socket
import numpy as np
import winsound
from imutils import face_utils
from scipy.spatial import distance as dist
from time import time, sleep
from pushbullet import Pushbullet
from custom_logging import text_log
from datetime import datetime

text_log(message='Imported libraries',
         show_console=True)

# GLOBAL VARIABLES ----------------------------

try:
    with open('.git/apikey.txt', 'r') as f:
        API = str(f.read())

    PB = Pushbullet(API)

except FileNotFoundError:
    text_log(message='API key not found',
             show_console=True)

except Exception as e:
    text_log(message=f'Error while connecting to API -> {e}',
             show_console=True)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 150)
GREEN = (0, 150, 0)

VID = cv2.VideoCapture(0)
# VID.set(cv2.CAP_PROP_SETTINGS, 0)
# VID.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
# VID.set(cv2.CAP_PROP_CONTRAST, 0.5)
# VID.set(cv2.CAP_PROP_EXPOSURE, 0.5)

YAWN_THRESH = 35
BLINK_THRESH = 0.145
SCORE_THRESH = 70
PAD = 50

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
HAAR_DATA = cv2.CascadeClassifier('data/frontfacedata.xml')
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Debugging
DEBUG_FACE = False
DEBUG_EAR = True
DEBUG_EYES = True
DEBUG_LIPS = True

# FLEET MANAGEMENT
DRIVER_ID = 'ATHTRS20'
VEHICLE_ID = 'MH12-AT-XXXX (Bolero-Pickup-4X4-White)'
ROUTE_ID = 'NH42-NH48'

text_log(message=f'Fleet Details: Driver {DRIVER_ID} Vehicle {VEHICLE_ID} Route {ROUTE_ID}')

text_log(message='Initialized variables',
         show_console=True)


def is_connected() -> bool:
    try:
        socket.create_connection(("www.google.com", 80))
        text_log(message='Internet connected')
        return True

    except Exception as ee:
        text_log(message=f'No internet connection available -> {ee}')
        return False


def notify(heading, message) -> None:

    try:
        push = PB.push_note(title=heading, body=message)
        text_log(message=f'Sent notification {heading} -> {message}')

    except Exception as ee:
        text_log(message=f'Failed to send notfication {ee}')


def exit_sequence() -> None:

    notify(heading=f'Driver {DRIVER_ID}',
           message='Quitting/Pausing Duty')

    cv2.destroyAllWindows()
    del DETECTOR, PREDICTOR, HAAR_DATA, DEBUG_FACE, DEBUG_LIPS, DEBUG_EYES, FONT
    del PAD, SCORE_THRESH, BLINK_THRESH, YAWN_THRESH
    del API, PB, WHITE, BLACK, GREEN, RED
    text_log(message='Quit')
    quit('Quitting')


def find_face() -> list:

    ret, frame_ = VID.read()

    if ret:
        grayscale = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
        face_coordinates = HAAR_DATA.detectMultiScale(image=grayscale,
                                                      scaleFactor=2,
                                                      minNeighbors=5,
                                                      minSize=(50, 50)
                                                      )

        if len(face_coordinates) == 1:
            x, y, w, h = [each for each in face_coordinates[0]]
            h = h+PAD

            subframe = grayscale[y: y + h, x: x + w]

        else:
            x, y = 0, 0
            w = grayscale.shape[1]
            h = grayscale.shape[0]
            subframe = grayscale

        if DEBUG_FACE:
            cv2.imshow('Subframe', subframe)
            cv2.waitKey(1)

        return [frame_, subframe, [x, y, x+w, y+h]]


def add_text(winname, message, location=(35, 35), colour=(255, 255, 255), thick=2, size=1.0) -> None:
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


def eye_aspect_ratio(eye) -> float:

    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])

    ear = (a + b) / (2.0 * c)

    return round(ear, 2)


def final_ear(shape) -> list:

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0

    return [ear, leftEye, rightEye]


def lip_distance(shape) -> float:

    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    dst = abs(top_mean[1] - low_mean[1])

    return dst


def blink_yawn(sub_frame) -> list:

    blinked = False
    yawned = False
    lip = None
    lt_hull, rt_hull = None, None

    faces = DETECTOR(sub_frame)

    for face in faces:
        shape = PREDICTOR(sub_frame, face)
        shape = face_utils.shape_to_np(shape)

        # EYES -----------------------------------------------------------

        eye = final_ear(shape)
        ear = eye[0]
        # print(ear)
        lt_eye = eye[1]
        rt_eye = eye[2]

        if DEBUG_EYES:
            lt_hull = cv2.convexHull(lt_eye)
            rt_hull = cv2.convexHull(rt_eye)

            cv2.drawContours(image=sub_frame, contours=[lt_hull], contourIdx=-1, color=GREEN, thickness=2)
            cv2.drawContours(image=sub_frame, contours=[rt_hull], contourIdx=-1, color=GREEN, thickness=2)

            add_text(winname=sub_frame,
                     message=f'EAR {round(ear, 2)}',
                     colour=WHITE,
                     size=0.75)

        if ear < BLINK_THRESH:
            blinked = True

        # LIPS ---------------------------------------------------------

        distance = lip_distance(shape)

        lip = shape[48:60]

        if DEBUG_LIPS:

            cv2.drawContours(image=sub_frame,
                             contours=[lip],
                             contourIdx=-1,
                             color=WHITE,
                             thickness=5)

        if distance > YAWN_THRESH:
            yawned = True

        cv2.imshow('Eyes and Lips Debug', sub_frame)
        cv2.waitKey(1)

    return [blinked, yawned, lt_hull, rt_hull, lip]


text_log(message='Defined functions',
         show_console=True)
print("Internet is connected" if is_connected() else "Internet is not connected, push notifications may not work")

notify(heading=f'Driver - {DRIVER_ID}',
       message=f'Starting journey in {VEHICLE_ID} via route {ROUTE_ID}')

blink_ctr = 0
yawn_ctr = 0
temp_blink = 0
temp_yawn = 0
delta = 3
score = 100
adder = 0

init = time()

text_log(message='Initialized local cars, starting...')

while True:

    # face_data = find_face()
    frame, subframe, box = find_face()

    blink = blink_yawn(subframe)[0]
    yawn = blink_yawn(subframe)[1]

    if blink:
        blink_ctr += 1
        temp_blink += 1
        sleep(0.1)

    if yawn:
        temp_yawn += 1

    # EYES CLOSED ----------------------------------
    if (time() - init <= delta) and temp_yawn >= 25:
        if score >= 5:
            score -= 5

        yawn_ctr += 1
        temp_yawn = 0

        if score < SCORE_THRESH:
            notify(heading=f'Driver {DRIVER_ID}',
                   message=f'Attentiveness dropped to {score}%')

        text_log(f'Eyes closed for more than threshold {BLINK_THRESH} -> Attentiveness {score}%')
        winsound.PlaySound('data/beep.wav', winsound.SND_FILENAME)

    # YAWNING ---------------------------------------
    if (time() - init <= delta) and temp_blink >= 15:
        if score >= 10:
            score -= 10

        temp_blink = 0

        if score < SCORE_THRESH:
            notify(heading=f'Driver {DRIVER_ID}',
                   message=f'Attentiveness dropped to {score}%')

        text_log(f'Driver yawning -> Attentiveness {score}%')
        winsound.PlaySound('data/beep.wav', winsound.SND_FILENAME)

    # RESET TIME
    if time() - init >= delta:
        init = time()
        adder += 1

    # GAIN SCORE
    if adder % ((60/delta) * 10) == 0 and score < 100:
        score += 1

    if DEBUG_FACE:
        cv2.rectangle(img=frame,
                      pt1=(box[0], box[1]),
                      pt2=(box[2], box[3]),
                      color=WHITE)

    # ADD TEXT INFO
    add_text(winname=frame,
             message=f'Blinks {blink_ctr}',
             colour=BLACK,
             size=0.5,
             location=(35, 35))

    add_text(winname=frame,
             message=f'Yawns {yawn_ctr} Score {score}',
             colour=BLACK,
             size=0.5,
             location=(35, 70))

    add_text(winname=frame,
             message=f'Score {score}',
             colour=GREEN if score >= SCORE_THRESH else RED,
             size=0.5,
             location=(35, 105))

    if DEBUG_LIPS and DEBUG_EYES:
        add_text(winname=frame,
                 message=f'Adder {adder}, Temp Blink/Yawn {temp_blink, temp_yawn}',
                 size=0.5,
                 location=(35, 140))

    if DEBUG_FACE:
        cv2.rectangle(img=frame,
                      pt1=(box[0], box[1]),
                      pt2=(box[2], box[3]),
                      color=GREEN,
                      thickness=2)

    cv2.imshow('Frame', frame)
    cv2.waitKey(1)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

exit_sequence()
