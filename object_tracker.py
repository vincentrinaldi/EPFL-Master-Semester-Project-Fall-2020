import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

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

### Vincent
import math
import shutil
from shapely.geometry import Point, Polygon
from tensorflow import keras
from tensorflow.keras import backend as K
from deep_sort.track import TrackState
flags.DEFINE_integer('vid_len', -1, 'maximum number of frames to process per video')
###

def main(_argv):
    # Definition of the parameters
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

    ### Vincent
    def deepball_loss_function(y_true, y_pred):
        pass
    def deepball_precision(y_true, y_pred):
        pass
    customObj = {'deepball_loss_function': deepball_loss_function, 'deepball_precision': deepball_precision}
    deep_ball_model = keras.models.load_model('./deepballlocal.h5', custom_objects=customObj)
    #deep_ball_model.summary()
    ###

    # begin video capture
    ### Vincent
    #try:
    #    vid = cv2.VideoCapture(int(video_path))
    #except:
    #    vid = cv2.VideoCapture(video_path)

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

    max_width = max([int(vid_blue.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_mid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_white.get(cv2.CAP_PROP_FRAME_WIDTH))])
    max_height = max([int(vid_blue.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid_mid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid_white.get(cv2.CAP_PROP_FRAME_HEIGHT))])
    max_fps = max([int(vid_blue.get(cv2.CAP_PROP_FPS)), int(vid_mid.get(cv2.CAP_PROP_FPS)), int(vid_white.get(cv2.CAP_PROP_FPS))])

    final_vid_len = min([int(vid_blue.get(cv2.CAP_PROP_FRAME_COUNT)), int(vid_mid.get(cv2.CAP_PROP_FRAME_COUNT)), int(vid_white.get(cv2.CAP_PROP_FRAME_COUNT))])
    if FLAGS.vid_len != -1:
        final_vid_len = FLAGS.vid_len
    ###

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        ### Vincent
        width = max_width #int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = max_height #int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = max_fps #int(vid.get(cv2.CAP_PROP_FPS))
        ###
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    ### Vincent
    # video size is 1920x1080 for blue and mid & 1920x1088 for white
    pts_video_blue = np.float32([[278,98],[1920,98],[0,1038],[1920,1038]]) #Blue side 1
    pts_video_mid = np.float32([[0,98],[1920,98],[0,1038],[1920,1038]]) #Mid side 1
    pts_video_white = np.float32([[0,98],[1642,98],[0,1038],[1920,1038]]) #White side 1

    # create polygons delimiting each side of the field on video
    blue_coords_field_poly = [(pts_video_blue[0][0], pts_video_blue[0][1]), (0, 460), (pts_video_blue[2][0], pts_video_blue[2][1]), (pts_video_blue[3][0], pts_video_blue[3][1]), (pts_video_blue[1][0], pts_video_blue[1][1])]
    blue_side_field_poly = Polygon(blue_coords_field_poly)
    mid_coords_field_poly = [(pts_video_mid[0][0], pts_video_mid[0][1]), (pts_video_mid[2][0], pts_video_mid[2][1]), (pts_video_mid[3][0], pts_video_mid[3][1]), (pts_video_mid[1][0], pts_video_mid[1][1])]
    mid_side_field_poly = Polygon(mid_coords_field_poly)
    white_coords_field_poly = [(pts_video_white[0][0], pts_video_white[0][1]), (pts_video_white[2][0], pts_video_white[2][1]), (pts_video_white[3][0], pts_video_white[3][1]), (1920, 460), (pts_video_white[1][0], pts_video_white[1][1])]
    white_side_field_poly = Polygon(white_coords_field_poly)

    # img size is 498x321
    pts_map_blue = np.float32([[0,0],[190,0],[22,320],[142,320]]) #Blue side 1 2D
    pts_map_mid = np.float32([[142,0],[355,0],[190,320],[307,320]]) #Mid side 1 2D
    pts_map_white = np.float32([[307,0],[497,0],[355,320],[475,320]]) #White side 1 2D

    # create polygon for mid side camera angle on 2D map
    mid_coords_map_poly = [(pts_map_mid[0][0], pts_map_mid[0][1]), (pts_map_mid[2][0], pts_map_mid[2][1]), (pts_map_mid[3][0], pts_map_mid[3][1]), (pts_map_mid[1][0], pts_map_mid[1][1])]
    mid_side_map_poly = Polygon(mid_coords_map_poly)

    # compute each perspective transformation matrix for bird’s-eye view with video and image 4-points landmark areas
    matrix_blue = cv2.getPerspectiveTransform(pts_video_blue, pts_map_blue)
    matrix_mid = cv2.getPerspectiveTransform(pts_video_mid, pts_map_mid)
    matrix_white = cv2.getPerspectiveTransform(pts_video_white, pts_map_white)

    # create list of videos to process and matrices to use
    vid_list = [vid_blue, vid_mid, vid_white]
    matrix_list = [matrix_blue, matrix_mid, matrix_white]

    # create brand new folder to save frames locally
    shutil.rmtree("processing", ignore_errors = True)
    os.mkdir("processing")

    # associate a subfolder to each video
    for i in range(len(vid_list)):
        os.mkdir("processing/" + str(i+1))

    # switch camera on the angle that detects the ball (mid angle in priority)
    idx_next_displayed_frame = 1
    sngl_ball_detected_per_frame = [[None for i in range(3)] for j in range(final_vid_len)]
    tot_rec_points_per_frame = [[] for i in range(final_vid_len)]

    idx_next_temp_prev = None
    valid_frame_detection_in_a_row = 0

    first_ball_detection = False #Test
    ###

    for idx, vid in enumerate(vid_list): #Vincent

        backSub = cv2.createBackgroundSubtractorKNN() #Testing Phase

        pos_range_x = 75 #Test
        pos_range_y = 75 #Test
        prev_pos_x = None #Test
        prev_pos_y = None #Test

        for track in tracker.tracks: #Vincent
            track.state = TrackState.Deleted #Vincent

        frame_num = 0
        # while video is running
        while True:
            ### Vincent
            #return_value, frame = vid.read()
            #if return_value:
            #    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #    image = Image.fromarray(frame)
            #else:
            #    print('Video has ended or failed, try a different video format!')
            #    break

            if frame_num < final_vid_len:
                return_value, frame = vid.read()
                if return_value:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                else:
                    if final_vid_len > frame_num:
                        final_vid_len = frame_num
                    print('Video has ended earlier than expected or has failed!')
                    break
            else:
                print('Limit of frames to process for this video is reached!')
                break
            ###

            frame_num +=1
            print('Frame #:', frame_num, 'Vid idx:', idx) #Vincent

            frame_size = frame.shape[:2]

            ### Vincent
            # resize frame to greatest existing dimensions among all processed video
            if frame_size[0] != max_height or frame_size[1] != max_width:
                frame = cv2.resize(frame, (max_width, max_height))

            # initialize the list of recorded detections for the current frame
            points_info = []
            ###

            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

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

            # by default allow all classes in .names file
            #allowed_classes = list(class_names.values())

            # custom allowed classes (uncomment line below to customize tracker for only people)
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

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            #initialize color map
            ### Vincent
            #cmap = plt.get_cmap('tab20b')
            #colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

			# Background Substraction and HSV Conversion
            foreGroundMask = backSub.apply(frame)
            frame_backSub = cv2.bitwise_and(frame, frame, mask = foreGroundMask)
            save_path = "processing/" + str(idx) + "_" + str(frame_num) + ".jpg" #For Testing
            cv2.imwrite(save_path, frame_backSub) #For Testing
            #frame_to_mask = np.asarray(frame_backSub)
            #frame_to_mask = cv2.cvtColor(frame_backSub, cv2.COLOR_RGB2HSV)

            # Colors
            # Range of H values :
            #red = 165 to 179 & 0 to 14 (use mask = cv2.bitwise_or(mask1,mask2))
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

            # red mask range (BGR)
            ref_low = (17, 15, 75)
            ref_high = (50, 56, 200)

            # blue mask range (BGR)
            home_low = (43, 31, 4)
            home_high = (250, 88, 50)

            # white mask range (BGR)
            away_low = (187,169,112)
            away_high = (255,255,255)

            # prepare the frame to be masked
            frame_to_mask = np.asarray(frame)
            frame_to_mask = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # create the different masks
            mask_ref = cv2.inRange(frame_to_mask, ref_low, ref_high)
            mask_home = cv2.inRange(frame_to_mask, home_low, home_high)
            mask_away = cv2.inRange(frame_to_mask, away_low, away_high)

            # apply each mask on the frame to create three resulting masked frames
            frame_masked_ref = cv2.bitwise_and(frame_to_mask, frame_to_mask, mask = mask_ref)
            frame_masked_home = cv2.bitwise_and(frame_to_mask, frame_to_mask, mask = mask_home)
            frame_masked_away = cv2.bitwise_and(frame_to_mask, frame_to_mask, mask = mask_away)
            ###

            ### Vincent
            cmap = deep_ball_model.predict(np.array([cv2.resize(frame.astype(np.float32), (480,272))]), batch_size=1, verbose=0)
            cm = cmap[0,:,:,0]

            pos = np.unravel_index(np.argmax(cm, axis=None), cm.shape)
            y,x = pos

            scr = cm[y,x]
            print(scr) #Test

            cv2.rectangle(frame, (0,0), (200,50), (255,0,0), -1) #Test
            cv2.putText(frame, str(scr) + "-" + str(frame_num),(0, 30),0, 0.75, (0,0,0),2) #Test

            x = -1 if scr < 0.999 else x #Test (change 0.99995 to 0.999)

            ky, kx = 4 * frame_size[0]/272.0, 4 * frame_size[1]/480.0
            y,x = math.floor(ky * y), math.floor(kx * x)

            if not x < 0:
                curr_ball_loc = Point(x, y+12) #Test (change 16 to 12)
                in_blue_side_but_not_inside_blue_poly = idx == 0 and not curr_ball_loc.within(blue_side_field_poly)
                in_mid_side_but_not_inside_mid_poly = idx == 1 and not curr_ball_loc.within(mid_side_field_poly)
                in_white_side_but_not_inside_white_poly = idx == 2 and not curr_ball_loc.within(white_side_field_poly)
                invalid_ball_pos = in_blue_side_but_not_inside_blue_poly or in_mid_side_but_not_inside_mid_poly or in_white_side_but_not_inside_white_poly
                x = -1 if invalid_ball_pos else x

            #if x < 0:
            #    sngl_ball_detected_per_frame[frame_num-1][idx] = 0
            #else:
            if not x < 0: #Test
                print("*** Ball detected ***")
                cv2.rectangle(frame, (0,0), (200,50), (0,0,255), -1) #Test
                cv2.putText(frame, str(scr) + "-" + str(frame_num),(0, 30),0, 0.75, (0,0,0),2) #Test

                #sngl_ball_detected_per_frame[frame_num-1][idx] = scr
                #Test
                if scr < 0.99999:
                    sngl_ball_detected_per_frame[frame_num-1][idx] = 0
                else:
                    sngl_ball_detected_per_frame[frame_num-1][idx] = scr
                ###

                #ball_bbox = np.array([x-16, y-16, 32, 32], dtype='f') #Test
                #bboxes = np.vstack((bboxes, ball_bbox)) #Test
                #scores = np.append(scores, scr) #Test
                #names = np.append(names, "ball") #Test

                if (prev_pos_x == None and prev_pos_y == None) or not scr < 0.99999: #Test
                    print("*** Ball drawn ***") #Test
                    cv2.rectangle(frame, (0,0), (200,50), (0,255,0), -1) #Test
                    cv2.putText(frame, str(scr) + "-" + str(frame_num),(0, 30),0, 0.75, (0,0,0),2) #Test

                    cv2.circle(frame, (x, y), 12, (255,255,0), 2) #Test (change 16 to 12 and 5 to 2) #Test
                    x_img, y_img = transform_coordinates_from_3D_to_2D(matrix_list[idx], x, y) #Test
                    points_info.append((0, x_img, y_img, (255,255,0), idx)) #Test

                    pos_range_x = 75 #Test
                    pos_range_y = 75 #Test
                    prev_pos_x = x #Test
                    prev_pos_y = y #Test
                elif x > prev_pos_x - pos_range_x and x < prev_pos_x + pos_range_x and y > prev_pos_y - pos_range_y and y < prev_pos_y + pos_range_y: #Test
                    print("*** Ball drawn ***") #Test
                    cv2.rectangle(frame, (0,0), (200,50), (0,255,0), -1) #Test
                    cv2.putText(frame, str(scr) + "-" + str(frame_num),(0, 30),0, 0.75, (0,0,0),2) #Test

                    cv2.circle(frame, (x, y), 12, (255,255,0), 2) #Test (change 16 to 12 and 5 to 2) #Test
                    x_img, y_img = transform_coordinates_from_3D_to_2D(matrix_list[idx], x, y)
                    points_info.append((0, x_img, y_img, (255,255,0), idx)) #Test

                    pos_range_x = 75 #Test
                    pos_range_y = 75 #Test
                    prev_pos_x = x #Test
                    prev_pos_y = y #Test
                else:
                    pos_range_x += 75 #Test
                    pos_range_y += 75 #Test
            else: #Test
                sngl_ball_detected_per_frame[frame_num-1][idx] = 0 #Test
                pos_range_x += 75 #Test
                pos_range_y += 75 #Test
            ###

            # update tracks
            for track in tracker.tracks:

                ### Vincent
                # compute current bbox properties
                curr_bbox_width = int(track.to_tlbr()[2])-int(track.to_tlbr()[0])
                curr_bbox_height = int(track.to_tlbr()[3])-int(track.to_tlbr()[1])
                curr_bbox_x = (int(track.to_tlbr()[0])+int(track.to_tlbr()[2]))/2
                curr_bbox_y = int(track.to_tlbr()[3])
                curr_bbox_pos = Point(curr_bbox_x, curr_bbox_y)
                in_blue_side_but_not_inside_blue_poly = idx == 0 and not curr_bbox_pos.within(blue_side_field_poly)
                in_mid_side_but_not_inside_mid_poly = idx == 1 and not curr_bbox_pos.within(mid_side_field_poly)
                in_white_side_but_not_inside_white_poly = idx == 2 and not curr_bbox_pos.within(white_side_field_poly)
                bool_invalid_bbox_pos = in_blue_side_but_not_inside_blue_poly or in_mid_side_but_not_inside_mid_poly or in_white_side_but_not_inside_white_poly

                # compute boolean on current bbox properties
                bool_curr_bbox_properties = curr_bbox_width > 110 or curr_bbox_height > 170 or bool_invalid_bbox_pos
                ###

                if not track.is_confirmed() or track.time_since_update > 1 or bool_curr_bbox_properties: #Vincent
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()
                score = track.get_score() #Vincent

            # draw bbox on screen
                ### Vincent
                #color = colors[int(track.track_id) % len(colors)]
                #color = [i * 255 for i in color]

                crop_image_ref = frame_masked_ref[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                crop_image_home = frame_masked_home[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                crop_image_away = frame_masked_away[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

                ref_ratio = black_pixels_ratio(crop_image_ref)
                home_ratio = black_pixels_ratio(crop_image_home)
                away_ratio = black_pixels_ratio(crop_image_away)

                ratio_list = [ref_ratio, home_ratio, away_ratio]
                color_box = None
                color_text = None
                if class_name == 'ball':
                    color_box = (255,255,0)
                    color_text = (0,0,0)
                else:
                    if min(ratio_list) == ref_ratio:
                        color_box = (255,0,0)
                        color_text = (255,255,255)
                    elif min(ratio_list) == home_ratio:
                        color_box = (0,0,255)
                        color_text = (255,255,255)
                    else:
                        color_box = (255,255,255)
                        color_text = (0,0,0)
                ###

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_box, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id))+len(str("{:.7f}".format(score))))*17, int(bbox[1])), color_box, -1) #Vincent
                cv2.putText(frame, class_name + "-" + str(track.track_id) + "-" + str("{:.7f}".format(score)),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, color_text,2) #Vincent

                ### Vincent
                # take middle of bbox bottom edge for 3D to 2D coordinates transformation
                px_vid = (int(bbox[0])+int(bbox[2]))/2
                py_vid = int(bbox[3])
                px_img, py_img = transform_coordinates_from_3D_to_2D(matrix_list[idx], px_vid, py_vid)
                points_info.append((track.track_id, px_img, py_img, color_box, idx)) #Test
                ###

            # if enable info flag then print details about each track
                if FLAGS.info:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

            ### Vincent
            # write frame to corresponding folder
            save_path = "processing/" + str(idx+1) + "/" + str(frame_num) + ".jpg"
            cv2.imwrite(save_path, frame)

            # append recorded detections in frame to list of total recorded detections in all videos for the same time step
            for pi in points_info:
                if FLAGS.info:
                    print("2D Point - Tracker ID: {}, X coord: {}, Y coord: {}, RGB color code: {}, Video idx: {}".format(pi[0], pi[1], pi[2], pi[3], pi[4])) #Test
                if idx != 1 or pi[0] == 0:
                    curr_point = Point(pi[1], pi[2])
                    if not curr_point.within(mid_side_map_poly):
                        tot_rec_points_per_frame[frame_num-1].append(pi)
                else:
                    tot_rec_points_per_frame[frame_num-1].append(pi)
            ###

            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)

    for i in range(final_vid_len): #Vincent

        ### Vincent
        # select next frame to display on final video
        count = len([j for j in sngl_ball_detected_per_frame[i] if j > 0])
        if count > 0:
            idx_next_temp = sngl_ball_detected_per_frame[i].index(max(sngl_ball_detected_per_frame[i]))
            if idx_next_temp != idx_next_displayed_frame:
                if idx_next_temp_prev == None:
                    idx_next_temp_prev = idx_next_temp
                    valid_frame_detection_in_a_row += 1
                elif idx_next_temp == idx_next_temp_prev:
                    valid_frame_detection_in_a_row += 1
                else:
                    idx_next_temp_prev = None
                    valid_frame_detection_in_a_row = 0
            else:
                idx_next_temp_prev = None
                valid_frame_detection_in_a_row = 0
            if valid_frame_detection_in_a_row == 4: #Test (change 5 to 4)
                idx_next_temp_prev = None
                valid_frame_detection_in_a_row = 0
                idx_next_displayed_frame = idx_next_temp

                if not first_ball_detection: #Test
                    first_ball_detection = True #Test

            #if count > 1:
            #    idx_next_displayed_frame = 1
            #else:
            #    idx_next_displayed_frame = sngl_ball_detected_per_frame[i].index(max(sngl_ball_detected_per_frame[i]))
        else:
            idx_next_temp_prev = None
            valid_frame_detection_in_a_row = 0
        load_path = "processing/" + str(idx_next_displayed_frame+1) + "/" + str(i+1) + ".jpg"
        next_displayed_frame = cv2.imread(load_path)
        result = np.asarray(next_displayed_frame) #np.asarray(frame)
        result = cv2.cvtColor(next_displayed_frame, cv2.COLOR_RGB2BGR) #cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # draw the four points of video landmark area
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

        # load bird’s-eye view image and draw every landmark areas points and each detection 2D spatial representation
        bird_eye = cv2.imread("data/img/football_field.jpg")
        cv2.circle(bird_eye, (0, 180), 2, (0,255,255),cv2.FILLED)
        cv2.circle(bird_eye, (497, 180), 2, (0,255,255),cv2.FILLED)
        pts_map_list = [pts_map_blue, pts_map_mid, pts_map_white]
        for pts_map in pts_map_list:
            cv2.circle(bird_eye, (pts_map[0][0], pts_map[0][1]), 1, (0,0,255),cv2.FILLED) #Test (change 2 to 1)
            cv2.circle(bird_eye, (pts_map[1][0], pts_map[1][1]), 1, (0,255,0),cv2.FILLED) #Test (change 2 to 1)
            cv2.circle(bird_eye, (pts_map[2][0], pts_map[2][1]), 1, (255,0,0),cv2.FILLED) #Test (change 2 to 1)
            cv2.circle(bird_eye, (pts_map[3][0], pts_map[3][1]), 1, (255,0,255),cv2.FILLED) #Test (change 2 to 1)
        for pi in tot_rec_points_per_frame[i]:
            if first_ball_detection or pi[0] != 0: #Test
                if pi[0] == 0: #Test
                    if pi[4] == idx_next_displayed_frame: #Test
                        cv2.circle(bird_eye, (pi[1], pi[2]), 3, (pi[3][2],pi[3][1],pi[3][0]),cv2.FILLED) #Test
                else: #Test
                    cv2.circle(bird_eye, (pi[1], pi[2]), 3, (pi[3][2],pi[3][1],pi[3][0]),cv2.FILLED) #Test (change 5 to 3)

        # shrink bird's-eye view image
        scale_percent = 60
        new_width = int(bird_eye.shape[1] * scale_percent / 100)
        new_height = int(bird_eye.shape[0] * scale_percent / 100)
        bird_eye = cv2.resize(bird_eye, (new_width, new_height), interpolation = cv2.INTER_AREA)

        # plot bird's-eye view image
        result_height, result_width = result.shape[:2]
        bird_eye_height, bird_eye_width = bird_eye.shape[:2]
        result[ int(result_height-bird_eye_height-50):int(result_height-50) , int((result_width/2)-(bird_eye_width/2)):int((result_width/2)+(bird_eye_width/2)) ] = bird_eye
        ###

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

### Vincent
def black_pixels_ratio(crop_image):
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
    px_img = (matrix[0][0]*px_vid + matrix[0][1]*py_vid + matrix[0][2]) / ((matrix[2][0]*px_vid + matrix[2][1]*py_vid + matrix[2][2]))
    py_img = (matrix[1][0]*px_vid + matrix[1][1]*py_vid + matrix[1][2]) / ((matrix[2][0]*px_vid + matrix[2][1]*py_vid + matrix[2][2]))
    return int(px_img), int(py_img)
###

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
