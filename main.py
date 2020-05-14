"""People Counter."""

'''
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m person-detection-retail-0013/FP32/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
'''
import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def ssd_out(frame, result):
    """
    Parse SSD output.
    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    current_count = 0
    for obj in result[0][0]:
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count = current_count + 1
    return frame, current_count

def performance_counts(perf_count):
    """
    print information about layers of the model.
    :param perf_count: Dictionary consists of status of the layers.
    :return: None
    """
    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def connect_mqtt():
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    # Flag for the input image
    single_image_mode = False

    cur_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    # Initialize the Inference Engine
    infer_network = Network()

    # Load the network model into the IE
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1,
                                          cur_request_id, 
                                          args.cpu_extension)[1]
    
    
    # Checks for live feed
    if args.input == 'CAM':
        input_stream = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input

    # Checks for video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")
    global initial_w, initial_h, prob_threshold
    prob_threshold = args.prob_threshold
    initial_w = cap.get(3)
    initial_h = cap.get(4)

    while cap.isOpened():
        
        #Reading the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        # Pre-process the frame
        image = cv2.resize(frame, (w, h))
        
        # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        
        inf_start = time.time()
        
        # Perform inference on the frame
        infer_network.exec_net(cur_request_id, image)

        if infer_network.wait(cur_request_id) == 0:
            
            det_time = time.time() - inf_start
            
            result = infer_network.get_output(cur_request_id)
            #if args.perf_counts:
            perf_count = infer_network.performance_counter(cur_request_id)
            #performance_counts(perf_count)

            frame, current_count = ssd_out(frame, result)
            
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
            # When new person enters the video
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))

            # Person duration in the video is calculated
            if current_count < last_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration", json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count

            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()

        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
            
        #CLEAN UP!
        #cap.release()
        #cv2.destroyAllWindows()
        #client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    
    
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
