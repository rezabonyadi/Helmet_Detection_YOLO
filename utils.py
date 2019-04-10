import cv2 as cv
import numpy as np


# Draw the predicted bounding box
def draw_bounding_box(classId, conf, left, top, right, bottom, frame, classes):
    frame_count = 0
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])

    label_name,label_conf = label.split(':')
    if label_name == 'Helmet':
        cv.rectangle(frame, (left, top - round(1.5*label_size[1])), (left + round(1.5*label_size[0]), top + base_line),
                     (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
        frame_count+=1


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, conf_threshold, nms_threshold, classes):
    frameHeight = frame.shape[0]
    frame_width = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    count_person=0 # for counting the classes in this loop.
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        draw_bounding_box(classIds[i], confidences[i], left, top, left + width, top + height, frame, classes)

        my_class='Helmet'
        unknown_class = classes[classId]

        if my_class == unknown_class:
            count_person += 1

    if(count_person > 0):
        print('Helmet detected')
    else:
        print('No helmet')

