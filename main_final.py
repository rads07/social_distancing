import numpy as np
import time
import cv2
import math

# Process to select that the person is at high risk or low risk or safe
angle_factor = 0.8
H_zoom_factor = 1.2
FR =0
def dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

def T2S(T):
    S = abs(T/((1+T**2)**0.5))
    return S

def T2C(T):
    C = abs(1/((1+T**2)**0.5))
    return C

def isclose(p1,p2):

    c_d = dist(p1[2], p2[2])
    if(p1[1]<p2[1]):
        a_w = p1[0]
        a_h = p1[1]
    else:
        a_w = p2[0]
        a_h = p2[1]

    T = 0
    try:
        T=(p2[2][1]-p1[2][1])/(p2[2][0]-p1[2][0])
    except ZeroDivisionError:
        T = 1.633123935319537e+16
    S = T2S(T)
    C = T2C(T)
    d_hor = C*c_d
    d_ver = S*c_d
    vc_calib_hor = a_w*1.3
    vc_calib_ver = a_h*0.4*angle_factor
    c_calib_hor = a_w *1.7
    c_calib_ver = a_h*0.2*angle_factor
    # print(p1[2], p2[2],(vc_calib_hor,d_hor),(vc_calib_ver,d_ver))
    if (0<d_hor<vc_calib_hor and 0<d_ver<vc_calib_ver):
        return 1
    elif 0<d_hor<c_calib_hor and 0<d_ver<c_calib_ver:
        return 2
    else:
        return 0



# yolo model configurations
labelsPath = "./yolo_files/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath ="./yolo_files/yolov3.weights"
configPath = "./yolo_files/yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# to get layernames like conv64,conv72,relu etc
ln = net.getLayerNames()


# Video for testing
video_name = input('Enter video Name:')
video_path = './videos/' + video_name
cap = cv2.VideoCapture('./videos/video.mp4')

# some variables
writer= None

# if video frame exists
print()
if(cap.isOpened()==False):
  print("Radhika")
n=0
avg_per =0
avg=[]


while(cap.isOpened()):
    
    ret,image=cap.read()
    (H, W) = image.shape[:2]
    FW=W
    if(W<1075):
        FW = 1075
    FR = np.zeros((H+210,FW,3), np.uint8)

    col = (255,255,255)
    FH = H + 210
    FR[:] = col

    # to get layernames like conv64,conv72,relu etc
    
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("Frame Prediction Time : {:.6f} seconds".format(end - start))
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5 and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
    
    # for finding centers of each box and defining them 
    # which is at risk and which is not
    if len(idxs) > 0:
    
        status = []
        idf = idxs.flatten()
        close_pair = []
        s_close_pair = []
        center = []
        co_info = []

        for i in idf:
            
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cen = [int(x + w / 2), int(y + h / 2)]
            center.append(cen)
            # cv2.circle(frame, tuple(cen),1,(0,0,0),1)
            co_info.append([w, h, cen])
            status.append(0)
        for i in range(len(center)):
            for j in range(len(center)):
                g = isclose(co_info[i],co_info[j])

                if g == 1:

                    close_pair.append([center[i], center[j]])
                    status[i] = 1
                    status[j] = 1
                elif g == 2:
                    s_close_pair.append([center[i], center[j]])
                    if status[i] != 1:
                        status[i] = 2
                    if status[j] != 1:
                        status[j] = 2

        total_p = len(center)
        low_risk_p = status.count(2)
        high_risk_p = status.count(1)
        safe_p = status.count(0)
       
        risk = low_risk_p/2 + high_risk_p
        avg_safe_percent = safe_p/(risk+safe_p) * 100  
        avg.append(avg_safe_percent)
        total_avg = sum(avg)/len(avg)
        kk=0
        print(total_avg)
        
            
        for i in idf:
            cv2.line(FR,(0,H+1),(FW,H+1),(0,0,0),2)
            cv2.putText(FR, "Social Distancing Analyser wrt. COVID-19", (210, H+60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(FR, (20, H+80), (510, H+180), (100, 100, 100), 2)
            
            cv2.putText(FR, "-- YELLOW: CLOSE", (50, H+90+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 170, 170), 2)
            cv2.putText(FR, "--    RED: VERY CLOSE", (50, H+40+110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv2.putText(frame, "--    PINK: Pathway for Calibration", (50, 150),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,105,255), 1)

            cv2.rectangle(FR, (535, H+80), (1060, H+140+40), (100, 100, 100), 2)
            cv2.putText(FR, "Bounding box shows the level of risk to the person.", (545, H+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 0), 2)
            cv2.putText(FR, "-- DARK RED: HIGH RISK", (565, H+90+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 2)
            cv2.putText(FR, "--   ORANGE: LOW RISK", (565, H+150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 2)

            cv2.putText(FR, "--    GREEN: SAFE", (565, H+170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)

            
            tot_str = "TOTAL COUNT: " + str(total_p)
            high_str = "HIGH RISK COUNT: " + str(high_risk_p)
            low_str = "LOW RISK COUNT: " + str(low_risk_p)
            safe_str = "SAFE COUNT: " + str(safe_p)
            Avg_str = "AVERAGE PERCENT: " + str(int(total_avg))

            cv2.putText(FR, tot_str, (10, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(FR, safe_str, (200, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 170, 0), 2)
            cv2.putText(FR, low_str, (380, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)
            cv2.putText(FR, high_str, (630, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)
            cv2.putText(FR, Avg_str, (840, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 255), 2)

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if status[kk] == 1:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 150), 2)

            elif status[kk] == 0:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            else:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 120, 255), 2)

            kk += 1
        
        FR[0:H, 0:W] = image
        image = FR
        cv2.imshow('Social distancing analyser', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("op_"+video_name, fourcc, 30,
                                 (image.shape[1], image.shape[0]), True)

    writer.write(image)

print('Processing Done')
writer.release()
cap.release()
cv2.destroyAllWindows()
