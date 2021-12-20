import cv2
import mediapipe as mp
import time
import numpy as np

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, image = cap.read()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:

            mpDraw.draw_landmarks(image=image, landmark_list=faceLms, connections=mpFaceMesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mpDraw.draw_landmarks(image=image, landmark_list=faceLms, connections=mpFaceMesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mpDraw.draw_landmarks(image=image, landmark_list=faceLms, connections=mpFaceMesh.FACEMESH_IRISES,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

            #mpDraw.draw_landmarks(image, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
            #mpDraw.draw_landmarks(image, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                #print(lm)
                ih, iw, ic = image.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                print(id,x,y)

            #############################################################
            ########-Hintergrund-bilder-########
            # Extract and draw pose on plain white image
            h, w, c = image.shape  # get shape of original frame
            opImg = np.zeros([h, w, c])  # create blank image with original frame size
            opImg.fill(0)  # set Black background. put 0 if you want to make it black and 255 if you want it white

            # draw extracted pose on black white image
            mpDraw.draw_landmarks(image=opImg, landmark_list=faceLms, connections=mpFaceMesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mpDraw.draw_landmarks(image=opImg, landmark_list=faceLms, connections=mpFaceMesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            cv2.imshow('Extracted Face', opImg)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(1)