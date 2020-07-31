from argparse import ArgumentParser

import logging
from InputFeeder import InputFeeder
import os
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarks
from gaze_estimation import GazeEstimation
from head_pose_estimation import HPE
from mouse_controller import MouseController
import cv2
import imutils
import math


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", "--face", required=True, type=str,
                        help="Path to .xml file of Face Detection model.")
    parser.add_argument("-fld", "--landmarks", required=True, type=str,
                        help="Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headpose", required=True, type=str,
                        help="Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-ge", "--gazeestimation", required=True, type=str,
                        help="Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or enter cam for webcam")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for model to detect the face accurately from the video frame.")
    parser.add_argument("-debug", "--debug", required=False, type=str, nargs='+', default=[],
                        help="To debug each model's output visually, type the model name with comma seperated after --debug")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="linker libraries if have any")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Provide the target device: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable.")
    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame," 
                             "fd for Face Detection, fld for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
    return parser


def main(args):
    
    logger = logging.getLogger()
    previewFlags = args.previewFlags
    InputFeeder = None
    input_filepath = args.input

    if inputFilePath.lower()=="cam":
            InputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFilePath):
            logger.error("Unable to find specified video file")
            exit(1)
        InputFeeder = InputFeeder("video",inputFilePath)

    
    model_paths = {'Face_detection_model': args.face,
                   'Facial_landmarks_detection_model': args.landmarks,
                   'head_pose_estimation_model': args.headpose,
                   'gaze_estimation_model': args.gazeestimation}

    face_detection_model = FaceDetectionModel(model_name=model_paths['Face_detection_model'],
                                                        device=args.device, threshold=args.prob_threshold,
                                                        extensions=args.cpu_extension)
    face_detection_model.check_model()

    facial_landmarks_detection_model = FacialLandmarks(model_name=model_paths['Facial_landmarks_detection_model'],
                                                        device=args.device, extensions=args.cpu_extension)
    facial_landmarks_detection_model.check_model()

    gaze_estimation_model = GazeEstimation(model_name=model_paths['gaze_estimation_model'],
                                            device=args.device, extensions=args.cpu_extension)
    gaze_estimation_model.check_model()

    head_pose_estimation_model = HPE(model_name=model_paths['head_pose_estimation_model'],
                                        device=args.device, extensions=args.cpu_extension)

    head_pose_estimation_model.check_model()

    mouse_controller = MouseController('medium', 'fast')

    face_detection_model.load_model()
    logger.info("Face Detection Model Loaded...")
    facial_landmarks_detection_model.load_model()
    logger.info("Facial Landmarks Model Loaded...")
    head_pose_estimation_model.load_model()
    logger.info("Had Pose Estimation Model Loaded...")
    gaze_estimation_model.load_model()
    logger.info("Gaze Estimation Model Loaded...")
    InputFeeder.load_data()
    logger.info("Input Feeder Loaded...")
    
    counter = 0
    start_inf_time = time.time()
    logger.info("Start inferencing on input video.. ")
    for flag, frame in InputFeeder.next_batch():
        if not flag:
            break
        pressed_key = cv2.waitKey(60)
        counter = counter + 1
        face_image, face_coordinates = face_detection_model.predict(frame.copy())

        if face_coordinates == None:
            continue

        head_pose_estimation_model_output = head_pose_estimation_model.predict(face_image.copy())

        left_eye_image, right_eye_image, eye_coord = facial_landmarks_detection_model.predict(face_image.copy())

        mouse_coordinate, gaze_vector = gaze_estimation_model.predict(left_eye_image, right_eye_image,
                                                                             head_pose_estimation_model_output)

        if (not len(previewFlags)==0):
            preview_frame = frame.copy()
            if 'fd' in previewFlags:
                preview_frame = face_image
            if 'fld' in previewFlags:
                cv2.rectangle(face_image, (eye_coords[0][0]-10, eye_coords[0][1]-10), (eye_coords[0][2]+10, eye_coords[0][3]+10), (0,255,0), 3)
                cv2.rectangle(face_image, (eye_coords[1][0]-10, eye_coords[1][1]-10), (eye_coords[1][2]+10, eye_coords[1][3]+10), (0,255,0), 3)
                
            if 'hp' in previewFlags:
                cv2.putText(preview_frame, "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(head_pose_estimation_model_output[0],head_pose_estimation_model_output[1],head_pose_estimation_model_output[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)
            if 'ge' in previewFlags:
                x, y, w = int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160
                le =cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(le, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                re = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(re, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                face_image[eye_coords[0][1]:eye_coords[0][3],eye_coords[0][0]:eye_coords[0][2]] = le
                face_image[eye_coords[1][1]:eye_coords[1][3],eye_coords[1][0]:eye_coords[1][2]] = re
                
            cv2.imshow("visualization",cv2.resize(preview_frame,(500,500)))
        
        if frame_count%5==0:
            mouse_controller.move(mouse_coordinate[0],mouse_coordinate[1])    
        if key==27:
                break
    logger.info("VideoStream ended...")
    cv2.destroyAllWindows()
    InputFeeder.close()
     
    

if __name__ == '__main__':
    args = build_argparser().parse_args()
    main(args)