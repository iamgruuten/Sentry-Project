#This program is mainly for asteria project

import cv2, os, time, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pymongo
from datetime import datetime
import keyboard

import numpy as np
from face_alignment import FaceMaskDetection
from tools import model_restore_from_pb
import tensorflow

import zmq
import time
import sys
import base64

import random

from paho.mqtt import client as mqtt_client

import pymongo
from pymongo import MongoClient
import base64

cluster = MongoClient("mongodb+srv://dbUser:Asteria987@cluster0.plmvw.mongodb.net/Sentry?retryWrites=true&w=majority")
db = cluster["Sentry"]
coll = db["Images"]
studentNoCapture = "hard"

broker = 'broker.emqx.io'
portMQTT = 1883
topic = "UniqueSentryImagePassable"
client_id = "UniqueSentryImagePassable1"

myclient = pymongo.MongoClient("mongodb+srv://dbUser:Asteria987@cluster0.plmvw.mongodb.net/Sentry?retryWrites=true&w=majority")

mycol = myclient['Sentry']['Logs']

port = "1111"

#Opening port 2222 for pi

if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
print("Tensorflow version: ",tf.__version__)
tf.get_logger().setLevel('INFO')

img_format = {'png','jpg','bmp'}

def log(studentNo, temp, type):
    date = datetime.today().strftime('%d/%m/%Y')
    time = datetime.today().strftime('%H:%M:%S')

    if mycol.count_documents({ '_id': studentNo }) == 0:
        log = { 
            "_id": studentNo,
            "entry":
            [
                {
                'date':date,   
                'timeString': time,
                'temp': temp,
                'type': type
            }
            ]
        }

        mycol.insert(log)
    else:
        mydict = { 
            "$push":{"entry":{
                    'date':date,   
                    'timeString': time,
                    'temp': temp,
                    'type': type
            }}
        }

        mycol.update_one({"_id":studentNo}, mydict)

def video_init(camera_source=0,resolution="480",to_write=False,save_dir=None):
    
    writer = None
    resolution_dict = {"480":[480,640],"720":[720,1280],"1080":[1080,1920]}

    
    cap = cv2.VideoCapture(camera_source)

    
    if resolution in resolution_dict.keys():
        width = resolution_dict[resolution][1]
        height = resolution_dict[resolution][0]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    if to_write is True:
        
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_path = 'demo.avi'
        if save_dir is not None:
            save_path = os.path.join(save_dir,save_path)
        writer = cv2.VideoWriter(save_path, fourcc, 30, (int(width), int(height)))

    return cap,height,width,writer



def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, portMQTT)
    return client


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        global studentNoCapture
        studentNoCapture = msg.payload.decode()
        getImage(msg.payload.decode())
        print(f"Received {msg.payload.decode()} from {msg.topic} topic")

    client.subscribe(topic)
    client.on_message = on_message
    


def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_start()

def getImage(id):
    #=================================================================
    results = coll.find({"_id":id}) #CHANGE THE ID
    #=================================================================

    for result in results:
        image = (result["Images"]) #Base64 data

    x = len(image) #Number of pictures
    image = image[x-1] #Get latest picture

    print(image)

    image = str(image)

    image = image.replace("[", '')
    image = image.replace("'", '')

    image = bytes(image, 'utf-8')
    #=================================================================
    with open("fileName.png", "wb") as fh: #CHANGE THE FILENAME
        fh.write(base64.decodebytes(image))
        keyboard.press_and_release('s')

    #=================================================================

def stream(pb_path, node_dict,ref_dir,camera_source=0,resolution="480",to_write=False,save_dir=None):

    print("starting stream...")
    frame_count = 0
    FPS = "loading"
    face_mask_model_path = r'face_mask_detection.pb'
    margin = 40
    id2class = {0: 'Mask', 1: 'NoMask'}
    batch_size = 32
    threshold = 0.8

    
    cap,height,width,writer = video_init(camera_source=camera_source, resolution=resolution, to_write=to_write, save_dir=save_dir)
    print("Loading Face Mask Detection")

    #Load Face mask detection layer
    fmd = FaceMaskDetection(face_mask_model_path, margin, GPU_ratio=None)

    #Restore trained model
    print("Loaded Face Mask Detection")

    sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=None)
    print("Restore PB file")

    tf_input = tf_dict['input']
    tf_phase_train = tf_dict['phase_train']
    tf_embeddings = tf_dict['embeddings']
    model_shape = tf_input.shape
    print("The mode shape of face recognition:",model_shape)
    feed_dict = {tf_phase_train: False}
    if 'keep_prob' in tf_dict.keys():
        tf_keep_prob = tf_dict['keep_prob']
        feed_dict[tf_keep_prob] = 1.0

    
    d_t = time.time()
    paths = [file.path for file in os.scandir(ref_dir) if file.name[-3:] in img_format]
    len_ref_path = len(paths)
    if len_ref_path == 0:
        print("No images in ", ref_dir)
    else:
        ites = math.ceil(len_ref_path / batch_size)
        embeddings_ref = np.zeros([len_ref_path, tf_embeddings.shape[-1]], dtype=np.float32)

        for i in range(ites):
            num_start = i * batch_size
            num_end = np.minimum(num_start + batch_size, len_ref_path)

            batch_data_dim =[num_end - num_start]
            batch_data_dim.extend(model_shape[1:])
            batch_data = np.zeros(batch_data_dim,dtype=np.float32)

            for idx,path in enumerate(paths[num_start:num_end]):
                img = cv2.imread(path)
                if img is None:
                    print("read failed:",path)
                else:
                    img = cv2.resize(img,(model_shape[2],model_shape[1]))
                    img = img[:,:,::-1]
                    batch_data[idx] = img
            batch_data /= 255
            feed_dict[tf_input] = batch_data

            embeddings_ref[num_start:num_end] = sess.run(tf_embeddings,feed_dict=feed_dict)

        d_t = time.time() - d_t

    
    if len_ref_path > 0:
        with tf.Graph().as_default():
            tf_tar = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape[-1])
            tf_ref = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape)
            tf_dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_ref, tf_tar)), axis=1))
            
            config = tf.ConfigProto(log_device_placement=True,
                                    allow_soft_placement=True,
                                    )
            config.gpu_options.allow_growth = True
            sess_cal = tf.Session(config=config)
            sess_cal.run(tf.global_variables_initializer())

        feed_dict_2 = {tf_ref: embeddings_ref}
    
    run()

    context = zmq.Context()
    socket = context.socket(zmq.REQ)

    socket.connect("tcp://192.168.43.120:%s" % port)

    while(cap.isOpened()):
        ret, img = cap.read()

        if ret is True:
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb.astype(np.float32)
            img_rgb /= 255

            img_fd = cv2.resize(img_rgb, fmd.img_size)
            img_fd = np.expand_dims(img_fd, axis=0)

            bboxes, re_confidence, re_classes, re_mask_id = fmd.inference(img_fd, height, width)

            if len(bboxes) > 0:
                for num, bbox in enumerate(bboxes):
                    class_id = re_mask_id[num]
                    if class_id == 0:
                        color = (0, 255, 0)  
                    else:
                        color = (0, 0, 255)  
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
                    
                    name = ""
                    if len_ref_path > 0:
                        img_fr = img_rgb[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]  
                        img_fr = cv2.resize(img_fr, (model_shape[2], model_shape[1]))  
                        img_fr = np.expand_dims(img_fr, axis=0)  

                        feed_dict[tf_input] = img_fr
                        embeddings_tar = sess.run(tf_embeddings, feed_dict=feed_dict)
                        feed_dict_2[tf_tar] = embeddings_tar[0]
                        distance = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                        arg = np.argmin(distance)  

                        if distance[arg] < threshold:
                            name = paths[arg].split("\\")[-1].split(".")[0]
                    cv2.putText(img, "{},{}".format(id2class[class_id], name), (bbox[0] + 2, bbox[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
                    print(name + " with " + id2class[class_id])


                    if name != "" :
                        if name == "Ben Low":
                            name = studentNoCapture

                        socket.send_string(name + " with " + id2class[class_id])
                        temp = socket.recv()
                        log(name, temp, "Entry")
                    else:
                       socket.send_string("0")
                       temp = socket.recv()

            if frame_count == 0:
                t_start = time.time()

            frame_count += 1
            if frame_count >= 10:
                FPS = "FPS=%1f" % (10 / (time.time() - t_start))
                frame_count = 0

            
            cv2.imshow("Gantry Face Recognition", img)

            if writer is not None:
                writer.write(img)
            
            #key handling to capture face and cut
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                img = cv2.imread("fileName.png")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgb = img_rgb.astype(np.float32)
                img_rgb /= 255

                img_fd = cv2.resize(img_rgb, fmd.img_size)
                img_fd = np.expand_dims(img_fd, axis=0)

                bboxes, re_confidence, re_classes, re_mask_id = fmd.inference(img_fd, height, width)

                if len(bboxes) > 0:
                    img_temp = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
                    save_path = studentNoCapture + ".jpg"
                    
                    save_path = os.path.join(ref_dir,save_path)
                    cv2.imwrite(save_path,img_temp)
                    print("An image is saved to ",save_path)

        else:
            print("get images failed")
            break

    
    cap.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()


if __name__ == "__main__":
    camera_source = 0
    pb_path = r"C:\Users\User\Desktop\Asteria\real\Realtime_face_recognition\pb_model.pb"
    node_dict = {'input': 'input:0',
                 'keep_prob': 'keep_prob:0',
                 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0',
                 }

    
    ref_dir = r"C:\Users\User\Desktop\Asteria\real\Realtime_face_recognition\text_xxx"
    stream(pb_path, node_dict, ref_dir, camera_source=camera_source, resolution="720", to_write=False, save_dir=None)


