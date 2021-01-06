# os library imports
import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# tensorflow library imports
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow import keras
from tensorflow.keras import backend as K
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# core folder imports
import core.utils as utils
from core.yolov4 import filter_boxes
from core.config import cfg

# deep_sort folder imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.track import TrackState

# tools folder imports
from tools import generate_detections as gdet

# common libraries imports
import time
import cv2
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags, logging
from absl.flags import FLAGS
from PIL import Image
from shapely.geometry import Point, Polygon

# definition of the different flags
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_integer('vid_len', -1, 'maximum number of frames to process per video')

def main(_argv):

    # definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # load deepball keras model for ball detection
    def deepball_loss_function(y_true, y_pred):
        pass
    def deepball_precision(y_true, y_pred):
        pass
    customObj = {'deepball_loss_function': deepball_loss_function, 'deepball_precision': deepball_precision}
    deep_ball_model = keras.models.load_model('./deepballlocal.h5', custom_objects=customObj)
    # uncomment below line to display the architecture of the deepball neural network
    #deep_ball_model.summary()

    # begin video capture (TODO:Iteration)
    vid_args = video_path.split("+")
    try:
        vid_blue = cv2.VideoCapture(int(vid_args[0]))
    except:
        vid_blue = cv2.VideoCapture(vid_args[0])
    try:
        vid_mid = cv2.VideoCapture(int(vid_args[1]))
    except:
        vid_mid = cv2.VideoCapture(vid_args[1])
    try:
        vid_white = cv2.VideoCapture(int(vid_args[2]))
    except:
        vid_white = cv2.VideoCapture(vid_args[2])

    # definition of width height fps and frame length of output video (TODO:Iteration)
    max_width = max([int(vid_blue.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_mid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_white.get(cv2.CAP_PROP_FRAME_WIDTH))])
    max_height = max([int(vid_blue.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid_mid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid_white.get(cv2.CAP_PROP_FRAME_HEIGHT))])
    max_fps = max([int(vid_blue.get(cv2.CAP_PROP_FPS)), int(vid_mid.get(cv2.CAP_PROP_FPS)), int(vid_white.get(cv2.CAP_PROP_FPS))])
    final_vid_len = min([int(vid_blue.get(cv2.CAP_PROP_FRAME_COUNT)), int(vid_mid.get(cv2.CAP_PROP_FRAME_COUNT)), int(vid_white.get(cv2.CAP_PROP_FRAME_COUNT))])
    if FLAGS.vid_len != -1:
        final_vid_len = FLAGS.vid_len

    out = None
    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = max_width
        height = max_height
        fps = max_fps
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    # definition of 4-points landmark on video (TODO:Flag to define video landmark & Iteration)
    # video size is 1920x1080 for blue and mid side & 1920x1088 for white side
    pts_video_blue = np.float32([[278,98],[1920,98],[0,1038],[1920,1038]]) #Blue side 1
    pts_video_mid = np.float32([[0,98],[1920,98],[0,1038],[1920,1038]]) #Mid side 1
    pts_video_white = np.float32([[0,98],[1642,98],[0,1038],[1920,1038]]) #White side 1

    # create polygons delimiting each side of the field on video (TODO:Flag to set side and fifth point & Iteration)
    blue_coords_field_poly = [(pts_video_blue[0][0], pts_video_blue[0][1]), (0, 460), (pts_video_blue[2][0], pts_video_blue[2][1]), (pts_video_blue[3][0], pts_video_blue[3][1]), (pts_video_blue[1][0], pts_video_blue[1][1])]
    blue_side_field_poly = Polygon(blue_coords_field_poly)
    mid_coords_field_poly = [(pts_video_mid[0][0], pts_video_mid[0][1]), (pts_video_mid[2][0], pts_video_mid[2][1]), (pts_video_mid[3][0], pts_video_mid[3][1]), (pts_video_mid[1][0], pts_video_mid[1][1])]
    mid_side_field_poly = Polygon(mid_coords_field_poly)
    white_coords_field_poly = [(pts_video_white[0][0], pts_video_white[0][1]), (pts_video_white[2][0], pts_video_white[2][1]), (pts_video_white[3][0], pts_video_white[3][1]), (1920, 460), (pts_video_white[1][0], pts_video_white[1][1])]
    white_side_field_poly = Polygon(white_coords_field_poly)

    # definition of 4-points landmark on image (TODO:Flag to define image landmark & Iteration)
    # img size is 498x321
    pts_map_blue = np.float32([[0,0],[190,0],[22,320],[142,320]]) #Blue side 1 2D
    pts_map_mid = np.float32([[142,0],[355,0],[190,320],[307,320]]) #Mid side 1 2D
    pts_map_white = np.float32([[307,0],[497,0],[355,320],[475,320]]) #White side 1 2D

    # create polygon for mid side camera angle on 2D map (TODO:Iteration)
    mid_coords_map_poly = [(pts_map_mid[0][0], pts_map_mid[0][1]), (pts_map_mid[2][0], pts_map_mid[2][1]), (pts_map_mid[3][0], pts_map_mid[3][1]), (pts_map_mid[1][0], pts_map_mid[1][1])]
    mid_side_map_poly = Polygon(mid_coords_map_poly)

    # compute each perspective transformation matrix for bird’s-eye view with video and image 4-points landmark areas (TODO:Iteration)
    matrix_blue = cv2.getPerspectiveTransform(pts_video_blue, pts_map_blue)
    matrix_mid = cv2.getPerspectiveTransform(pts_video_mid, pts_map_mid)
    matrix_white = cv2.getPerspectiveTransform(pts_video_white, pts_map_white)

    # create list of videos to process and matrices to use (TODO:Iteration)
    vid_list = [vid_blue, vid_mid, vid_white]
    matrix_list = [matrix_blue, matrix_mid, matrix_white]

    # create brand new folder to save frames locally
    shutil.rmtree("processing", ignore_errors = True)
    os.mkdir("processing")

    # associate a subfolder to each video
    for i in range(len(vid_list)):
        os.mkdir("processing/" + str(i+1))

    # initialize variable to store index of video of next frame to display with mid angle as priority (TODO:Iteration)
    idx_next_displayed_frame = 1

    # store for each frame of each video the nearest bbox from the ball if the latter has been detected
    nearest_bbox_from_ball = [[None for i in range(3)] for j in range(final_vid_len)]

    # store the recorded position of the ball on video
    recorded_ball_positions = [[None for i in range(3)] for j in range(final_vid_len)]

    # store the number of times a frame has been designed to be the best one for each video
    count_best_frame_ball_detection = [[0 for i in range(3)] for j in range(int((final_vid_len - 1)/max_fps) + 1)]

    # store ball detection scores higher than true ball threshold (TODO: Iteration)
    sngl_ball_detected_per_frame = [[None for i in range(3)] for j in range(final_vid_len)]

    # initialize list of detections to draw as dots on 2D map for each selected frame
    tot_rec_points_per_frame = [[] for i in range(final_vid_len)]

    # start iteration on each video frame of every video
    for idx, vid in enumerate(vid_list):

        # initialize background subtractor that is using KNN algorithm
        backSub = cv2.createBackgroundSubtractorKNN()

        # initialize variables for validation of ball detection
        pos_range_x = 100
        pos_range_y = 100
        prev_pos_x = None
        prev_pos_y = None

        # Delete tracks of DeepSort tracker that were computed in previous video
        for track in tracker.tracks:
            track.state = TrackState.Deleted

        # initialize frame count
        frame_num = 0

        # while video is running
        while True:
            if frame_num < final_vid_len:
                return_value, frame = vid.read()
                if return_value:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    if final_vid_len > frame_num:
                        final_vid_len = frame_num
                    print('Video has ended earlier than expected or has failed!')
                    break
            else:
                print('Limit of frames to process for this video is reached!')
                break

            # start timer for current frame processing
            start_time = time.time()

            # increment frame count
            frame_num +=1
            print('Frame #:', frame_num, 'Vid idx:', idx)

            # get frame size in the form [height, width]
            frame_size = frame.shape[:2]

            # resize frame to greatest existing dimensions among all processed video
            if frame_size[0] != max_height or frame_size[1] != max_width:
                frame = cv2.resize(frame, (max_width, max_height))

            # initialize the list of recorded detections and bbox coordinates for the current frame
            points_info = []
            bboxs_info = []

            """
            ####################################################################
            Run YOLOv4 detector and DeepSort tracker
            ####################################################################
            """

            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            # run detections on tflite if flag is set
            if FLAGS.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                # run detections using yolov3 if flag is set
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # allow all classes in .names file or custom ones to customize tracker
            # uncomment line below to allow all classes
            #allowed_classes = list(class_names.values())
            # uncomment line below to allow only custom classes
            allowed_classes = ['person']

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)
            if FLAGS.count:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                print("Objects being tracked: {}".format(count))

            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # call the tracker
            tracker.predict()
            tracker.update(detections)

            """
            ####################################################################
            """

			# apply background subtraction on current frame
            foreGroundMask = backSub.apply(frame)
            frame_backSub = cv2.bitwise_and(frame, frame, mask = foreGroundMask)

            # switch masked frame from RGB to BGR (or HSV) mode
            #frame_to_mask = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #TOREMOVE
            #frame_to_mask = cv2.cvtColor(frame_backSub, cv2.COLOR_RGB2BGR) #Test
            frame_to_mask = cv2.cvtColor(frame_backSub, cv2.COLOR_RGB2HSV) #Test

            #"""
            # Colors
            # Range of H values :
            #red = 165 to 179 & 0 to 14
            #yellow = 15 to 44
            #green = 45 to 74
            #cyan = 75 to 104
            #blue = 105 to 134
            #magenta = 135 to 164
            # S values are 25 to 255
            # V values are 125 to 255

            # Non colors
            # H values are 0 to 255
            # Range of S and V values :
            #white = 0 to 24 // 125 to 255
            #black = 0 to 255 // 0 to 124

            # red mask range (HSV)
            #ref_low_1 = (0, 25, 125)
            #ref_low_1 = (0, 76, 76)
            ref_low_1 = (0, 100, 100)
            #ref_high_1 = (14, 255, 255)
            ref_high_1 = (9, 255, 255)
            #ref_low_2 = (165, 25, 125)
            #ref_low_2 = (165, 76, 76)
            ref_low_2 = (170, 100, 100)
            ref_high_2 = (179, 255, 255)

            # blue mask range (HSV)
            #home_low = (105, 25, 125)
            #home_low = (105, 76, 76)
            home_low = (110, 75, 75)
            #home_high = (134, 255, 255)
            home_high = (130, 255, 255)

            # white mask range (HSV)
            #away_low = (0,0,125)
            #away_low = (0,0,180)
            away_low = (0,0,180)
            #away_high = (255,24,255)
            #away_high = (179,75,255)
            away_high = (179,75,255)

            mask_ref_1 = cv2.inRange(frame_to_mask, ref_low_1, ref_high_1)
            mask_ref_2 = cv2.inRange(frame_to_mask, ref_low_2, ref_high_2)
            mask_ref = cv2.bitwise_or(mask_ref_1, mask_ref_2)
            mask_home = cv2.inRange(frame_to_mask, home_low, home_high)
            mask_away = cv2.inRange(frame_to_mask, away_low, away_high)

            # home goalie mask range (HSV) #Test
            home_goalie_low = (25,100,100)
            home_goalie_high = (35,255,255)
            mask_home_goalie = cv2.inRange(frame_to_mask, home_goalie_low, home_goalie_high)
            frame_masked_home_goalie = cv2.bitwise_and(frame_to_mask, frame_to_mask, mask = mask_home_goalie)

            # away goalie mask range (HSV) #Test
            away_goalie_low = (40,20,120)
            away_goalie_high = (90,80,230)
            mask_away_goalie = cv2.inRange(frame_to_mask, away_goalie_low, away_goalie_high)
            frame_masked_away_goalie = cv2.bitwise_and(frame_to_mask, frame_to_mask, mask = mask_away_goalie)

            """

            # red mask range (BGR)
            #ref_low = (17, 15, 75)
            #ref_high = (50, 56, 200)
            #ref_low = (16, 16, 96)
            #ref_high = (31, 31, 255)
            ref_low = (16, 16, 72)
            ref_high = (47, 55, 200)

            # blue mask range (BGR)
            #home_low = (43, 31, 4)
            #home_high = (250, 88, 50)
            #home_low = (48, 32, 32)
            #home_high = (255, 47, 47)
            home_low = (40, 32, 8)
            home_high = (247, 87, 47)

            # white mask range (BGR)
            #away_low = (187,169,112)
            #away_high = (255,255,255)
            #away_low = (224,224,224)
            #away_high = (255,255,255)
            away_low = (184,168,112)
            away_high = (255,255,255)

            # create the different masks for referees, home and away teams jersey
            mask_ref = cv2.inRange(frame_to_mask, ref_low, ref_high)
            mask_home = cv2.inRange(frame_to_mask, home_low, home_high)
            mask_away = cv2.inRange(frame_to_mask, away_low, away_high)
            """

            # apply each mask on the same current frame to create three resulting masked frames
            frame_masked_ref = cv2.bitwise_and(frame_to_mask, frame_to_mask, mask = mask_ref)
            frame_masked_home = cv2.bitwise_and(frame_to_mask, frame_to_mask, mask = mask_home)
            frame_masked_away = cv2.bitwise_and(frame_to_mask, frame_to_mask, mask = mask_away)

            """
            ####################################################################
            Run DeepBall detector
            ####################################################################
            """

            ball_position = None

            # predict ball detection confidence score on every pixel of the frame
            #cmap = deep_ball_model.predict(np.array([cv2.resize(frame.astype(np.float32), (480,272))]), batch_size=1, verbose=0) #TOREMOVE
            cmap = deep_ball_model.predict(np.array([cv2.resize(frame_backSub.astype(np.float32), (480,272))]), batch_size=1, verbose=0)
            cm = cmap[0,:,:,0]

            # retrieve pixel position having highest ball detection confidence score
            pos = np.unravel_index(np.argmax(cm, axis=None), cm.shape)
            y,x = pos

            # retrieve highest ball detection confidence score value
            scr = cm[y,x]
            #print(scr) #TOREMOVE
            cv2.rectangle(frame, (0,0), (700,50), (255,0,0), -1)
            cv2.putText(frame, str(scr) + "-" + str(frame_num), (10, 30), 0, 0.75, (0,0,0), 2)

            # check if highest ball detection confidence score is less than minimum acceptable threshold
            x = -1 if scr < 0.999999 else x

            # compute coordinates of highest ball detection confidence score pixel on full size frame
            ky, kx = 4 * frame_size[0]/272.0, 4 * frame_size[1]/480.0
            y,x = math.floor(ky * y), math.floor(kx * x)

            # if position is valid we check if this is a true ball detection or if new position of the ball is consistent
            if not x < 0:
                print("*** Ball detected ***")
                cv2.rectangle(frame, (0,0), (700,50), (0,0,255), -1)
                cv2.putText(frame, str(scr) + "-" + str(frame_num), (10, 30), 0, 0.75, (255,255,255), 2)

                sngl_ball_detected_per_frame[frame_num-1][idx] = scr

                # check if current ball detection is consistent
                # if this is a valid ball detection we reset the range in addition to updating the previous valid ball detection position
                if (prev_pos_x == None and prev_pos_y == None):
                    print("*** Ball drawn ***")
                    cv2.rectangle(frame, (0,0), (700,50), (0,255,0), -1)
                    cv2.putText(frame, str(scr) + "-" + str(frame_num), (10, 30), 0, 0.75, (0,0,0), 2)

                    cv2.circle(frame, (x, y), 12, (255,255,0), 2)
                    cv2.circle(frame_backSub, (x, y), 12, (255,255,0), 2)
                    # save masked frame with background subtraction
                    save_path = "processing/" + str(idx) + "_" + str(frame_num) + ".jpg"
                    cv2.imwrite(save_path, frame_backSub)

                    x_img, y_img = transform_coordinates_from_3D_to_2D(matrix_list[idx], x, y+12)
                    points_info.append((0, x_img, y_img, (255,255,0), idx))

                    ball_position = (x, y)

                    pos_range_x = 100
                    pos_range_y = 100
                    prev_pos_x = x
                    prev_pos_y = y
                # if the ball detection is in a certain range from the previous valid ball detection then we consider that it is a valid ball detection and reset the range in addition to updating the previous valid ball detection position
                elif x > prev_pos_x - pos_range_x and x < prev_pos_x + pos_range_x and y > prev_pos_y - pos_range_y and y < prev_pos_y + pos_range_y:
                    print("*** Ball drawn ***")
                    cv2.rectangle(frame, (0,0), (700,50), (0,255,0), -1)
                    cv2.putText(frame, str(scr) + "-" + str(frame_num), (10, 30), 0, 0.75, (0,0,0), 2)

                    cv2.circle(frame, (x, y), 12, (255,255,0), 2)
                    cv2.circle(frame_backSub, (x, y), 12, (255,255,0), 2)
                    # save masked frame with background subtraction
                    save_path = "processing/" + str(idx) + "_" + str(frame_num) + ".jpg"
                    cv2.imwrite(save_path, frame_backSub)

                    x_img, y_img = transform_coordinates_from_3D_to_2D(matrix_list[idx], x, y+12)
                    points_info.append((0, x_img, y_img, (255,255,0), idx))

                    ball_position = (x, y)

                    pos_range_x = 100
                    pos_range_y = 100
                    prev_pos_x = x
                    prev_pos_y = y
                # otherwise it's not a consistent next position and we increase the range where the ball can be from the last valid detection position
                else:
                    pos_range_x += 100
                    pos_range_y += 100
            # otherwise we increase the range where the ball can be from the last valid detection position
            else:
                sngl_ball_detected_per_frame[frame_num-1][idx] = 0
                pos_range_x += 100
                pos_range_y += 100

            # if we are processing the last video we fill the array of ordered video index from which to display the frame on the final video
            if idx == 2:
                max_rec = -1
                idx_max = []
                for i in range(3):
                    temp = sngl_ball_detected_per_frame[frame_num-1][i]
                    if temp > max_rec:
                        max_rec = temp
                        idx_max = [i]
                    elif temp == max_rec:
                        idx_max.append(i)
                if max_rec != 0:
                    for i in idx_max:
                        count_best_frame_ball_detection[int((frame_num - 1)/max_fps)][i] += 1

            """
            ####################################################################
            """

            # update tracks
            for track in tracker.tracks:

                # compute current bbox dimensions
                curr_bbox_width = int(track.to_tlbr()[2])-int(track.to_tlbr()[0])
                curr_bbox_height = int(track.to_tlbr()[3])-int(track.to_tlbr()[1])

                # compute current bbox position
                curr_bbox_x = (int(track.to_tlbr()[0])+int(track.to_tlbr()[2]))/2
                curr_bbox_y = int(track.to_tlbr()[3])
                curr_bbox_pos = Point(curr_bbox_x, curr_bbox_y)

                # check validity of current bbox position
                in_blue_side_but_not_inside_blue_poly = idx == 0 and not curr_bbox_pos.within(blue_side_field_poly)
                in_mid_side_but_not_inside_mid_poly = idx == 1 and not curr_bbox_pos.within(mid_side_field_poly)
                in_white_side_but_not_inside_white_poly = idx == 2 and not curr_bbox_pos.within(white_side_field_poly)
                bool_invalid_bbox_pos = in_blue_side_but_not_inside_blue_poly or in_mid_side_but_not_inside_mid_poly or in_white_side_but_not_inside_white_poly

                # compute boolean gathering validity of dimensions and position of current bbox
                bool_curr_bbox_properties = curr_bbox_width > 110 or curr_bbox_height > 170 or bool_invalid_bbox_pos

                if not track.is_confirmed() or track.time_since_update > 1 or bool_curr_bbox_properties:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()

                # crop each masked frames to only keep current bbox area
                crop_image_ref = frame_masked_ref[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                crop_image_home = frame_masked_home[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                crop_image_away = frame_masked_away[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

                # compute black pixels ratio of current bbox area on each masked frames
                ref_ratio = black_pixels_ratio(crop_image_ref)
                home_ratio = black_pixels_ratio(crop_image_home)
                away_ratio = black_pixels_ratio(crop_image_away)

                #Test
                crop_image_home_goalie = frame_masked_home_goalie[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                home_goalie_ratio = black_pixels_ratio(crop_image_home_goalie)
                crop_image_away_goalie = frame_masked_away_goalie[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                away_goalie_ratio = black_pixels_ratio(crop_image_away_goalie)

                # create list gathering every ratio result
                #ratio_list = [ref_ratio, home_ratio, away_ratio]
                ratio_list = [ref_ratio, home_ratio, away_ratio, home_goalie_ratio, away_goalie_ratio] #Test
                #print(track.track_id, ratio_list) #Test

                # compute displayed color of current bbox and its text by choosing the lowest ratio of black pixels
                color_box = None
                color_text = None
                if min(ratio_list) == ref_ratio:
                    color_box = (255,0,0)
                    color_text = (255,255,255)
                elif min(ratio_list) == home_ratio or min(ratio_list) == home_goalie_ratio: #Test
                    color_box = (0,0,255)
                    color_text = (255,255,255)
                else:
                    color_box = (255,255,255)
                    color_text = (0,0,0)

                # draw current bbox on screen
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_box, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(str(track.track_id)))*17, int(bbox[1])), color_box, -1)
                cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, color_text, 2)

                # take middle of bbox bottom edge for 3D to 2D coordinates transformation
                px_vid = (int(bbox[0])+int(bbox[2]))/2
                py_vid = int(bbox[3])
                px_img, py_img = transform_coordinates_from_3D_to_2D(matrix_list[idx], px_vid, py_vid)

                # add recorded detection to list gathering detections for current frame
                points_info.append((track.track_id, px_img, py_img, color_box, idx))

                # add bbox coordinates and track id to list that will be used for statistics computation
                bboxs_info.append((track.track_id, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), color_box))

                # if enable info flag then print details about each track
                if FLAGS.info:
                    print("Tracker ID: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

            # write frame to corresponding folder
            save_path = "processing/" + str(idx+1) + "/" + str(frame_num) + ".jpg"
            cv2.imwrite(save_path, frame)

            # append all recorded detections on frame to list gathering detections for frames at same time step of each video if condition is met
            for pi in points_info:
                if FLAGS.info:
                    print("2D Point - Tracker ID: {}, X coord: {}, Y coord: {}, RGB color code: {}, Video idx: {}".format(pi[0], pi[1], pi[2], pi[3], pi[4]))
                # if it's not the ball and not detected on mid camera angle then if it's inside mid polygon we don't add it to list (TODO:Iteration)
                if idx != 1 and pi[0] != 0:
                    curr_point = Point(pi[1], pi[2])
                    if not curr_point.within(mid_side_map_poly):
                        tot_rec_points_per_frame[frame_num-1].append(pi)
                else:
                    tot_rec_points_per_frame[frame_num-1].append(pi)

            if ball_position != None:
                recorded_ball_positions[frame_num-1][idx] = ball_position
                min_dist = 999999
                selected_bbox_idx = -1
                for bi_idx, bi in enumerate(bboxs_info):
                    if bi[5] != (255,0,0):
                        center_bbox = Point((bi[1]+bi[3])/2, (bi[2]+bi[4])/2)
                        ball_point = Point(ball_position[0], ball_position[1])
                        distance_to_center_bbox = ball_point.distance(center_bbox)
                        #print(bi) #TOREMOVE
                        #print(center_bbox) #TOREMOVE
                        #print(ball_point) #TOREMOVE
                        #print(bi[0], distance_to_center_bbox) #TOREMOVE
                        if min_dist > distance_to_center_bbox:
                            min_dist = distance_to_center_bbox
                            selected_bbox_idx = bi_idx
                #print(selected_bbox_idx) #TOREMOVE
                if selected_bbox_idx != -1:
                    selected_bbox = bboxs_info[selected_bbox_idx]
                    nearest_bbox_from_ball[frame_num-1][idx] = (selected_bbox[0], min_dist, selected_bbox[5])

            # display time taken to run detections on the current frame in terms of fps
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)

    print(count_best_frame_ball_detection) #TOREMOVE

    nb_possession_frames = 0
    possession_home = 0
    possession_away = 0

    prev_possession = None
    currently_passing = False
    total_passes_home = 0
    correct_passes_home = 0
    total_passes_away = 0
    correct_passes_away = 0

    polygons_list = [blue_side_field_poly, mid_side_field_poly, white_side_field_poly]

    heatmap_values = [[0 for i in range(498)] for j in range(321)]

    # start computation of output video
    for i in range(final_vid_len):

        next_idx = [j for j, x in enumerate(count_best_frame_ball_detection[int(i/max_fps)]) if x == max(count_best_frame_ball_detection[int(i/max_fps)])]
        if len(next_idx) == 1:
            idx_next_displayed_frame = next_idx[0]

        # load previously corresponding saved frame
        load_path = "processing/" + str(idx_next_displayed_frame+1) + "/" + str(i+1) + ".jpg"
        next_displayed_frame = cv2.imread(load_path)

        # switch loaded frame from RGB to BGR mode
        result = cv2.cvtColor(next_displayed_frame, cv2.COLOR_RGB2BGR) #cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # draw the corresponding 4-points landmark area on output video frame (TODO:Iteration)
        pts_displayed_frame = None
        if idx_next_displayed_frame == 0:
            cv2.circle(result, (0, 460), 2, (0,255,255),cv2.FILLED)
            pts_displayed_frame = pts_video_blue
        elif idx_next_displayed_frame == 1:
            pts_displayed_frame = pts_video_mid
        else:
            cv2.circle(result, (1920, 460), 2, (0,255,255),cv2.FILLED)
            pts_displayed_frame = pts_video_white
        cv2.circle(result, (pts_displayed_frame[0][0], pts_displayed_frame[0][1]), 2, (0,0,255),cv2.FILLED)
        cv2.circle(result, (pts_displayed_frame[1][0], pts_displayed_frame[1][1]), 2, (0,255,0),cv2.FILLED)
        cv2.circle(result, (pts_displayed_frame[2][0], pts_displayed_frame[2][1]), 2, (255,0,0),cv2.FILLED)
        cv2.circle(result, (pts_displayed_frame[3][0], pts_displayed_frame[3][1]), 2, (255,0,255),cv2.FILLED)

        # update the different statistics
        # statistics for possession and passing accuracy which needs to ball to be detected for the current frame
        if nearest_bbox_from_ball[i][idx_next_displayed_frame] != None:
            # check which player from which team is the closest to the ball
            color_rectangle = None
            color_text = None
            team = None
            if nearest_bbox_from_ball[i][idx_next_displayed_frame][2] == (0,0,255):
                color_rectangle = (255,0,0)
                color_text = (255,255,255)
                team = "Home"
            else:
                color_rectangle = (255,255,255)
                color_text = (0,0,0)
                team = "Away"

            # check if the ball is too far from the player meaning we are in a passing phase or if the ball is very close to him meaning it's possession phase
            phase = None
            if nearest_bbox_from_ball[i][idx_next_displayed_frame][1] > 70:
                phase = "Passing"
                currently_passing = True
            else:
                phase = "Possession"
                # if it was a passing phase at the previous frame we will check if the ball was the closest to an other player in the previous possession phase
                if currently_passing:
                    currently_passing = False
                    if prev_possession != None:
                        # we can only do that if we are on the same camera angle
                        if idx_next_displayed_frame == prev_possession[2]:
                            # if it was a different player before then we update the passing accuracy statistics
                            if nearest_bbox_from_ball[i][idx_next_displayed_frame][0] != prev_possession[0]:
                                # if the previous player was from the same team then it's a successful pass
                                if team == prev_possession[1]:
                                    if team == "Home":
                                        total_passes_home += 1
                                        correct_passes_home += 1
                                    else:
                                        total_passes_away += 1
                                        correct_passes_away += 1
                                # otherwise the player missed the pass
                                else:
                                    if team == "Home":
                                        total_passes_away += 1
                                    else:
                                        total_passes_home += 1
                        # otherwise we just check the team that had the ball at the previous possession phase
                        else:
                            # if the previous player was from the same team then it's a successful pass
                            if team == prev_possession[1]:
                                if team == "Home":
                                    total_passes_home += 1
                                    correct_passes_home += 1
                                else:
                                    total_passes_away += 1
                                    correct_passes_away += 1
                            # otherwise the player missed the pass
                            else:
                                if team == "Home":
                                    total_passes_away += 1
                                else:
                                    total_passes_home += 1

                # we update the current nearest player as being the previous reference player of the next frame to process
                prev_possession = (nearest_bbox_from_ball[i][idx_next_displayed_frame][0], team, idx_next_displayed_frame)

                # we update the possession statistics
                nb_possession_frames += 1
                if team == "Home":
                    possession_home += 1
                else:
                    possession_away += 1

            # we display the phase information on the screen
            cv2.rectangle(result, (1220,0), (1920,50), color_rectangle, -1)
            cv2.putText(result, str(nearest_bbox_from_ball[i][idx_next_displayed_frame][0]) + " - " + phase, (1230, 30), 0, 0.75, color_text, 2)
        else:
            cv2.rectangle(result, (1220,0), (1920,50), (0,0,0), -1)
            cv2.putText(result, "No Valid Ball Detected", (1230, 30), 0, 0.75, (255,255,255), 2)

        # we display the possession statistics information on the screen
        cv2.rectangle(result, (1220,1038), (1920,1088), (0,0,0), -1)
        ratio_possession_home = str(possession_home) + "/" + str(nb_possession_frames)
        ratio_possession_away = str(possession_away) + "/" + str(nb_possession_frames)
        if nb_possession_frames > 0:
            cv2.putText(result, str(int((possession_home/nb_possession_frames)*100)) + "%" + " (" + ratio_possession_home + ")" + " - Possession - " + str(int((possession_away/nb_possession_frames)*100)) + "%" + " (" + ratio_possession_away + ")", (1230, 1068), 0, 0.75, (255,255,255), 2)
        else:
            cv2.putText(result, str(0) + "%" + " (" + ratio_possession_home + ")" + " - Possession - " + str(0) + "%" + " (" + ratio_possession_away + ")", (1230, 1068), 0, 0.75, (255,255,255), 2)

        # we display the passing accuracy statistics information on the screen
        cv2.rectangle(result, (0,1038), (700,1088), (255,255,255), -1)
        pass_accuracy_home = 0
        pass_accuracy_away = 0
        if total_passes_home > 0:
            pass_accuracy_home = int((correct_passes_home/total_passes_home)*100)
        if total_passes_away > 0:
            pass_accuracy_away = int((correct_passes_away/total_passes_away)*100)
        ratio_pass_home = str(correct_passes_home) + "/" + str(total_passes_home)
        ratio_pass_away = str(correct_passes_away) + "/" + str(total_passes_away)
        cv2.putText(result, str(pass_accuracy_home) + "%" + " (" + ratio_pass_home + ")" + " - Pass Accuracy - " + str(pass_accuracy_away) + "%" + " (" + ratio_pass_away + ")", (10, 1068), 0, 0.75, (0,0,0), 2)

        # we display the current phase of the play according to ball position
        if recorded_ball_positions[i][idx_next_displayed_frame] != None:
            ball_pos = recorded_ball_positions[i][idx_next_displayed_frame]
            ball_pt = Point(ball_pos[0], ball_pos[1])
            play_phase = None
            if not ball_pt.within(polygons_list[idx_next_displayed_frame]):
                play_phase = "Out of Bounds"
            else:
                play_phase = "On Field"
            if play_phase == "Out of Bounds":
                cv2.rectangle(result, (810,0), (1110,50), (0,0,255), -1)
                cv2.putText(result, play_phase, (820, 30), 0, 0.75, (0,0,0), 2)
            else:
                cv2.rectangle(result, (810,0), (1110,50), (0,255,0), -1)
                cv2.putText(result, play_phase, (820, 30), 0, 0.75, (0,0,0), 2)
        else:
            cv2.rectangle(result, (810,0), (1110,50), (0,0,0), -1)
            cv2.putText(result, "No Ball Detected", (820, 30), 0, 0.75, (255,255,255), 2)

        # load bird’s-eye view image and draw all landmark areas on 2D map
        bird_eye = cv2.imread("data/img/football_field.jpg")
        cv2.circle(bird_eye, (0, 180), 2, (0,255,255),cv2.FILLED)
        cv2.circle(bird_eye, (497, 180), 2, (0,255,255),cv2.FILLED)
        pts_map_list = [pts_map_blue, pts_map_mid, pts_map_white]
        for pts_map in pts_map_list:
            cv2.circle(bird_eye, (pts_map[0][0], pts_map[0][1]), 1, (0,0,255),cv2.FILLED)
            cv2.circle(bird_eye, (pts_map[1][0], pts_map[1][1]), 1, (0,255,0),cv2.FILLED)
            cv2.circle(bird_eye, (pts_map[2][0], pts_map[2][1]), 1, (255,0,0),cv2.FILLED)
            cv2.circle(bird_eye, (pts_map[3][0], pts_map[3][1]), 1, (255,0,255),cv2.FILLED)
        # draw all valid detections recorded on every frame at current time step of each video on 2D map
        for pi in tot_rec_points_per_frame[i]:
            if pi[0] == 0:
                if pi[4] == idx_next_displayed_frame:
                    cv2.circle(bird_eye, (pi[1], pi[2]), 3, (pi[3][2],pi[3][1],pi[3][0]),cv2.FILLED)
            else:
                cv2.circle(bird_eye, (pi[1], pi[2]), 3, (pi[3][2],pi[3][1],pi[3][0]),cv2.FILLED)
                if pi[3][2] == 255 and pi[3][1] == 0 and pi[3][0] == 0: #TODO:Flag
                    heatmap_values[pi[2]][pi[1]] += 1 #Test

        # shrink bird's-eye view image (TODO:Flag)
        scale_percent = 60
        new_width = int(bird_eye.shape[1] * scale_percent / 100)
        new_height = int(bird_eye.shape[0] * scale_percent / 100)
        bird_eye = cv2.resize(bird_eye, (new_width, new_height), interpolation = cv2.INTER_AREA)

        # plot bird's-eye view image on output frame
        result_height, result_width = result.shape[:2]
        bird_eye_height, bird_eye_width = bird_eye.shape[:2]
        result[ int(result_height-bird_eye_height-50):int(result_height-50) , int((result_width/2)-(bird_eye_width/2)):int((result_width/2)+(bird_eye_width/2)) ] = bird_eye

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set we save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    heatmap_background = cv2.imread("data/img/football_field.jpg")
    for i in range(heatmap_background.shape[0]):
        for j in range(heatmap_background.shape[1]):
            heatmap_values[i][j] = int((float(heatmap_values[i][j]) / float(final_vid_len)) * 255)
    heatmap_values = np.array(heatmap_values)
    heatmap_values = heatmap_values.astype(np.uint8)
    heatmap_values = cv2.applyColorMap(heatmap_values, cv2.COLORMAP_JET)
    # write heatmap to img folder
    #save_path = "data/img/heatmap_player_" + str(1) + ".jpg"
    #cv2.imwrite(save_path, heatmap_values)

    heatmap_values = cv2.cvtColor(heatmap_values, cv2.COLOR_BGR2BGRA)
    for i in range(heatmap_values.shape[0]):
        for j in range(heatmap_values.shape[1]):
            if heatmap_values[i][j][0] == 128 and heatmap_values[i][j][1] == 0 and heatmap_values[i][j][2] == 0:
                heatmap_values[i][j][3] = 0

    #save_path = "data/img/heatmap_player_" + str(1) + "_transparency.png"
    #cv2.imwrite(save_path, heatmap_values)

    heatmap_background = cv2.cvtColor(heatmap_background, cv2.COLOR_BGR2BGRA)

    # normalize alpha channels from 0-255 to 0-1
    alpha_background = heatmap_background[:,:,3] / 255.0
    alpha_foreground = heatmap_values[:,:,3] / 255.0

    # set adjusted colors
    for color in range(0, 3):
        heatmap_background[:,:,color] = alpha_foreground * heatmap_values[:,:,color] + alpha_background * heatmap_background[:,:,color] * (1 - alpha_foreground)

    # set adjusted alpha and denormalize back to 0-255
    heatmap_background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

    save_path = "data/img/heatmap_player_blue_final.png"
    cv2.imwrite(save_path, heatmap_background)

    cv2.destroyAllWindows()

### Vincent
def black_pixels_ratio(crop_image):
    """
    Count the black pixel ratio on current masked bbox area

    Parameters
    ----------
    crop_image : the current masked bbox area

    Returns
    -------
    count/(height*width) : The black pixels ratio
    """
    # get frame height and width to access pixels
    height, width, _ = crop_image.shape
    if height == 0 or width == 0:
        return 1.0

    # access BGR pixel values on each channel
    count = 0
    for x in range(0, width) :
         for y in range(0, height) :
             if crop_image[y,x,0] == 0 and crop_image[y,x,1] == 0 and crop_image[y,x,2] == 0:
                count += 1

    return count/(height*width)

def transform_coordinates_from_3D_to_2D(matrix, px_vid, py_vid):
    """
    Convert coordinates of detection from video to image

    Parameters
    ----------
    matrix : the transformation matrix
    px_vid : the x coordinate on video
    py_vid : the y coordinate on video

    Returns
    -------
    int(px_img), int(py_img) : The x and y coordinates on image
    """
    px_img = (matrix[0][0]*px_vid + matrix[0][1]*py_vid + matrix[0][2]) / ((matrix[2][0]*px_vid + matrix[2][1]*py_vid + matrix[2][2]))
    py_img = (matrix[1][0]*px_vid + matrix[1][1]*py_vid + matrix[1][2]) / ((matrix[2][0]*px_vid + matrix[2][1]*py_vid + matrix[2][2]))
    return int(px_img), int(py_img)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
