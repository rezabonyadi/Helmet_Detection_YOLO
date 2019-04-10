from time import sleep

# [model](https://drive.google.com/open?id=16yH9M_ovw0cJG4gVKuXTkz_cwYxJtwAk)
# [cfg](https://drive.google.com/open?id=1GiWyY1EHUWgkBvo8tGuwM4yoplaZZGza)

from utils import postprocess
import cv2 as cv
frame_count = 0             # used in mainloop  where we're extracting images., and then to drawPred( called by post process)
frame_count_out=0           # used in post process loop, to get the no of specified class value.
# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image


# Load names of classes
classesFile = "obj.names"

with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3-obj.cfg";
modelWeights = "yolov3-obj_2400.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
layersNames = net.getLayerNames()
# Get the names of the output layers, i.e. the layers with unconnected outputs
output_layer = [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# # cap = VideoCapture(0)
cap = cv.VideoCapture(0)
# # for fn in glob('images/*.jpg'):
while True:
    ret, frame = cap.read()
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(output_layer)

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs, confThreshold, nmsThreshold, classes)
    cv.imshow('img', frame)
    t, _ = net.getPerfProfile()
    #print(t)
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    #print(label)
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    #print(label)
