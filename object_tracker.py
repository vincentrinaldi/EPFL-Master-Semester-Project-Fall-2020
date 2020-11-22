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

### Vincent
from tensorflow import keras
from tensorflow.keras import backend as K
import math
###

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
    customObj = {'deepball_loss_function': deepball_loss_function, 'deepball_precision': deepball_precision}
    deep_ball_model = keras.models.load_model('./deepballlocal.h5', custom_objects=customObj)
    ###

    # begin video capture
    #try: #Vincent
    #    vid = cv2.VideoCapture(int(video_path)) #Vincent
    #except: #Vincent
    #    vid = cv2.VideoCapture(video_path) #Vincent

    ### Vincent
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
    frame_iter_len = max_fps * 15
    nb_full_frame_iter = final_vid_len // frame_iter_len
    last_frame_iter_len = final_vid_len % frame_iter_len
    frame_iter_process_list = []
    if last_frame_iter_len < max_fps * 5:
        frame_iter_process_list = [frame_iter_len for i in range(nb_full_frame_iter - 1)]
        frame_iter_process_list.append(last_frame_iter_len + frame_iter_len)
    else:
        frame_iter_process_list = [frame_iter_len for i in range(nb_full_frame_iter)]
        frame_iter_process_list.append(last_frame_iter_len)
    ###

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        #width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) #Vincent
        #height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) #Vincent
        #fps = int(vid.get(cv2.CAP_PROP_FPS)) #Vincent
        width = max_width #Vincent
        height = max_height #Vincent
        fps = max_fps #Vincent
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    ### Vincent
    # video size is 1920x1080 for blue and mid & 1920x1088 for white
    pts_video_blue = np.float32([[278,92],[1920,92],[0,1038],[1920,1038]]) #Blue side 1
    pts_video_mid = np.float32([[0,90],[1920,90],[0,1028],[1920,1028]]) #Mid side 1
    pts_video_white = np.float32([[0,102],[1642,102],[0,1032],[1920,1032]]) #White side 1

    # img size is 498x321
    pts_map_blue = np.float32([[0,0],[190,0],[22,320],[142,320]]) #Blue side 1 2D
    pts_map_mid = np.float32([[142,0],[355,0],[190,320],[307,320]]) #Mid side 1 2D
    pts_map_white = np.float32([[307,0],[497,0],[355,320],[475,320]]) #White side 1 2D

    # computation of perspective transformation matrix for bird’s-eye view with video and image 4-points landmark areas
    matrix_blue = cv2.getPerspectiveTransform(pts_video_blue, pts_map_blue)
    matrix_mid = cv2.getPerspectiveTransform(pts_video_mid, pts_map_mid)
    matrix_white = cv2.getPerspectiveTransform(pts_video_white, pts_map_white)
    ###

    # switch camera on angle detecting the ball (mid angle in priority)
    vid_list = [vid_blue, vid_mid, vid_white] #Vincent
    matrix_list = [matrix_blue, matrix_mid, matrix_white] #Vincent
    quit_pressed = False #Vincent
    for nb_frames_to_process in frame_iter_process_list: #Vincent
        min_nb_valid_frames = nb_frames_to_process
        sngl_frame_state_per_frame = [[None for i in range(3)] for j in range(nb_frames_to_process)] #Vincent
        #sngl_nb_detections_per_frame = [[None for i in range(3)] for j in range(nb_frames_to_process)] #Vincent
        sngl_ball_detected_per_frame = [[None for i in range(3)] for j in range(nb_frames_to_process)] #Vincent
        tot_rec_points_per_frame = [None for i in range(nb_frames_to_process)] #Vincent
        for idx, vid in enumerate(vid_list): #Vincent

            frame_num = 0
            # while video is running
            while True:
                #return_value, frame = vid.read() #Vincent
                #if return_value: #Vincent
                #    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Vincent
                #    image = Image.fromarray(frame) #Vincent
                #else: #Vincent
                #    print('Video has ended or failed, try a different video format!') #Vincent
                #    break #Vincent

                ### Vincent
                if frame_num < nb_frames_to_process:
                    return_value, frame = vid.read()
                    if return_value:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame)
                    else:
                        if min_nb_valid_frames > frame_num:
                            min_nb_valid_frames = frame_num
                        print('Video has ended earlier than expected or has failed, try a different video format!')
                        break
                else:
                    print('Limit of frames to process at once is reached, processing next batch of frames of subsequent video!')
                    break
                ###

                frame_num +=1
                print('Frame #: ', frame_num)

                frame_size = frame.shape[:2]
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
                allowed_classes = ['person', 'sports ball']

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

                #initialize color map
                #cmap = plt.get_cmap('tab20b') #Vincent
                #colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)] #Vincent

                ### Vincent
                ref_low = (17, 15, 75) #red (BGR)
                ref_high = (50, 56, 200)
                home_low = (43, 31, 4) #blue (BGR)
                home_high = (250, 88, 50)
                away_low = (187,169,112) #white (BGR)
                away_high = (255,255,255)
                frame_to_mask = np.asarray(frame)
                frame_to_mask = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                mask_ref = cv2.inRange(frame_to_mask, ref_low, ref_high)
                mask_home = cv2.inRange(frame_to_mask, home_low, home_high)
                mask_away = cv2.inRange(frame_to_mask, away_low, away_high)
                frame_masked_ref = cv2.bitwise_and(frame_to_mask, frame_to_mask, mask = mask_ref)
                frame_masked_home = cv2.bitwise_and(frame_to_mask, frame_to_mask, mask = mask_home)
                frame_masked_away = cv2.bitwise_and(frame_to_mask, frame_to_mask, mask = mask_away)
                ###

                # run non-maxima supression
                boxs = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                classes = np.array([d.class_name for d in detections])
                indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                points_info = [] #Vincent

                # update tracks
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1 or int(track.to_tlbr()[2])-int(track.to_tlbr()[0]) > 110 or int(track.to_tlbr()[3])-int(track.to_tlbr()[1]) > 170: #Vincent
                        continue
                    bbox = track.to_tlbr()
                    class_name = track.get_class()
                    score = track.get_score() #Vincent

                # draw bbox on screen
                    #color = colors[int(track.track_id) % len(colors)] #Vincent
                    #color = [i * 255 for i in color] #Vincent

                    crop_image_ref = frame_masked_ref[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    crop_image_home = frame_masked_home[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    crop_image_away = frame_masked_away[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

                    ref_ratio = black_pixels_ratio(crop_image_ref)
                    home_ratio = black_pixels_ratio(crop_image_home)
                    away_ratio = black_pixels_ratio(crop_image_away)

                    ratio_list = [ref_ratio, home_ratio, away_ratio]
                    color_box = None
                    color_text = None
                    if class_name == 'sports ball':
                        color_box = (255,255,0)
                        color_text = (0,0,0)
                    elif min(ratio_list) == ref_ratio:
                        color_box = (255,0,0)
                        color_text = (255,255,255)
                    elif min(ratio_list) == home_ratio:
                        color_box = (0,0,255)
                        color_text = (255,255,255)
                    else:
                        color_box = (255,255,255)
                        color_text = (0,0,0)

                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_box, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id))+len(str("{:.2f}".format(score))))*17, int(bbox[1])), color_box, -1) #Vincent
                    cv2.putText(frame, class_name + "-" + str(track.track_id) + "-" + str("{:.2f}".format(score)),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, color_text,2) #Vincent

                    ### Vincent
                    px_vid = (int(bbox[0])+int(bbox[2]))/2
                    py_vid = int(bbox[3])
                    px_img, py_img = transform_coordinates_from_3D_to_2D(matrix_list[idx], px_vid, py_vid)
                    points_info.append((track.track_id, px_img, py_img, color_box))
                    ###

                # if enable info flag then print details about each track
                    if FLAGS.info:
                        print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

                ### Vincent
                sngl_frame_state_per_frame[frame_num-1][idx] = frame
                #sngl_nb_detections_per_frame[frame_num-1][idx] = len(points_info)
                if idx == 0:
                    tot_rec_points_per_frame[frame_num-1] = []
                for pi in points_info:
                    if FLAGS.info:
                        print("2D Point - Tracker ID: {}, X coord: {}, Y coord: {}, RGB color code: {}".format(pi[0], pi[1], pi[2], pi[3]))
                    tot_rec_points_per_frame[frame_num-1].append(pi)
                ###

                ### Vincent
                ky, kx = 4 * frame_size[0]/272.0, 4 * frame_size[1]/480.0
                cmap = deep_ball_model.predict(np.array([cv2.resize(frame.astype(np.float32), (480,272))]), batch_size=1, verbose=1)
                cm = cmap[0,:,:,0]
                pos = np.unravel_index(np.argmax(cm, axis=None), cm.shape)
                y,x = pos
                x = -1 if cm[y,x] < 0.999999 else x
                y,x = math.floor(ky * y), math.floor(kx * x)
                if x < 0:
                    print("*** No ball detected ***")
                    sngl_ball_detected_per_frame[frame_num-1][idx] = 0
                else:
                    cv2.circle(frame, (x, y), 16, (255,255,0), 2)
                    sngl_ball_detected_per_frame[frame_num-1][idx] = 1
                ###

                # calculate frames per second of running detections
                fps = 1.0 / (time.time() - start_time)
                print("FPS: %.2f" % fps)

        for i in range(min_nb_valid_frames): #Vincent
            #idx_next_displayed_frame = sngl_nb_detections_per_frame[i].index(max(sngl_nb_detections_per_frame[i])) #Vincent
            count = len([j for j in sngl_ball_detected_per_frame[i] if j > 0]) #Vincent
            if count > 1: #Vincent
                idx_next_displayed_frame = 1 #Vincent
            else: #Vincent
                idx_next_displayed_frame = sngl_ball_detected_per_frame[i].index(max(sngl_ball_detected_per_frame[i])) #Vincent
            next_displayed_frame = sngl_frame_state_per_frame[i][idx_next_displayed_frame] #Vincent

            #result = np.asarray(frame) #Vincent
            #result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #Vincent

            result = np.asarray(next_displayed_frame) #Vincent
            result = cv2.cvtColor(next_displayed_frame, cv2.COLOR_RGB2BGR) #Vincent

            ### Vincent
            pts_displayed_frame = None
            if idx_next_displayed_frame == 0:
                pts_displayed_frame = pts_video_blue
            elif idx_next_displayed_frame == 1:
                pts_displayed_frame = pts_video_mid
            else:
                pts_displayed_frame = pts_video_white
            cv2.circle(result, (pts_displayed_frame[0][0], pts_displayed_frame[0][1]), 2, (0,0,255),cv2.FILLED)
            cv2.circle(result, (pts_displayed_frame[1][0], pts_displayed_frame[1][1]), 2, (0,255,0),cv2.FILLED)
            cv2.circle(result, (pts_displayed_frame[2][0], pts_displayed_frame[2][1]), 2, (255,0,0),cv2.FILLED)
            cv2.circle(result, (pts_displayed_frame[3][0], pts_displayed_frame[3][1]), 2, (255,0,255),cv2.FILLED)

            # plotting bird’s-eye view
            bird_eye = cv2.imread("data/img/football_field.jpg")
            pts_map_list = [pts_map_blue, pts_map_mid, pts_map_white]
            for pts_map in pts_map_list:
                cv2.circle(bird_eye, (pts_map[0][0], pts_map[0][1]), 2, (0,0,255),cv2.FILLED)
                cv2.circle(bird_eye, (pts_map[1][0], pts_map[1][1]), 2, (0,255,0),cv2.FILLED)
                cv2.circle(bird_eye, (pts_map[2][0], pts_map[2][1]), 2, (255,0,0),cv2.FILLED)
                cv2.circle(bird_eye, (pts_map[3][0], pts_map[3][1]), 2, (255,0,255),cv2.FILLED)
            for pi in tot_rec_points_per_frame[i]:
                cv2.circle(bird_eye, (pi[1], pi[2]), 5, (pi[3][2],pi[3][1],pi[3][0]),cv2.FILLED)

            # shrinking bird eye view image
            scale_percent = 60
            new_width = int(bird_eye.shape[1] * scale_percent / 100)
            new_height = int(bird_eye.shape[0] * scale_percent / 100)
            bird_eye = cv2.resize(bird_eye, (new_width, new_height), interpolation = cv2.INTER_AREA)

            result_height, result_width = result.shape[:2]
            bird_eye_height, bird_eye_width = bird_eye.shape[:2]
            result[ int(result_height-bird_eye_height-50):int(result_height-50) , int((result_width/2)-(bird_eye_width/2)):int((result_width/2)+(bird_eye_width/2)) ] = bird_eye

            # adapting final frame dimensions to output writer
            if result_height != max_height or result_width != max_width:
                result = cv2.resize(result, (max_width, max_height))
            ###

            if not FLAGS.dont_show:
                cv2.imshow("Output Video", result)

            # if output flag is set, save video file
            if FLAGS.output:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                quit_pressed = True
                break
        if quit_pressed:
            break
    cv2.destroyAllWindows()

### Vincent
def black_pixels_ratio(crop_image):
    #Get frame height and width to access pixels
    height, width, _ = crop_image.shape
    if height == 0 or width == 0:
        return 1.0
    #Accessing BGR pixel values
    count = 0
    for x in range(0, width) :
         for y in range(0, height) :
             if crop_image[y,x,0] == 0 and crop_image[y,x,1] == 0 and crop_image[y,x,2] == 0: #BGR Channel Value
                count += 1
    return count/(height*width)

def transform_coordinates_from_3D_to_2D(matrix, px_vid, py_vid):
    px_img = (matrix[0][0]*px_vid + matrix[0][1]*py_vid + matrix[0][2]) / ((matrix[2][0]*px_vid + matrix[2][1]*py_vid + matrix[2][2]))
    py_img = (matrix[1][0]*px_vid + matrix[1][1]*py_vid + matrix[1][2]) / ((matrix[2][0]*px_vid + matrix[2][1]*py_vid + matrix[2][2]))
    return int(px_img), int(py_img)
###

### Vincent
def deepball_loss_function(y_true, y_pred):
    # y_true (batch_size, 68, 120, 2)
    # y_pred (batch_size, 68, 120, 2)

    ball_gt, bg_gt = y_true[:,:,:,0], y_true[:,:,:,1]
    N = K.sum(ball_gt, axis=(1,2)) + 1
    M = K.sum(bg_gt, axis=(1,2)) + 1
    zer = K.zeros_like(ball_gt)

    y_pred = K.log(y_pred)
    ball_cm = y_pred[:,:,:,0]
    bg_cm = y_pred[:,:,:,1]

    Lpos = K.sum(zer + (ball_cm * ball_gt), axis=(1,2))
    Lpos = K.sum(K.zeros_like(N) + (Lpos / tf.maximum(1.0, N)))

    Lneg = K.sum(zer + (bg_cm * bg_gt), axis=(1,2))
    Lneg = K.sum(K.zeros_like(M) + (Lneg / tf.maximum(1.0, M)))
    #print(K.eval(Lpos),K.eval(Lneg))

    # Multiplying by batch_size as Keras automatically averages the scalar output over it
    return (-Lpos - 0.3*Lneg) * 16

def deepball_precision(y_true, y_pred):
    ball_gt = y_true[:,:,:,0]
    ball_cm = y_pred[:,:,:,0]

    thre_ball_cm = K.cast(K.greater(ball_cm, 0.998), "float32")
    tp = K.sum(ball_gt * thre_ball_cm)
    totalp = K.sum(K.max(thre_ball_cm, axis=(1,2)))

    return tp/tf.maximum(1.0, totalp)
###

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
