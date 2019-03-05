# 1. Experiment with edge detection
# 2. Corner Detection: Translation, Rotation, but not scaling
# 3. Scale-invariant Feature Transformation: Translation, Rotation AND not scaling
# 4. KeyPoints and matching

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cur_mode = None


def sift(img):
    sift = cv2.xfeatures2d.SIFT_create()
    img_2 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints = sift.detect(gray, None)

    cv2.drawKeypoints(img_2, keypoints, img_2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img_2

def contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _,contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_2 = img.copy()
    img_con = cv2.drawContours(img_2, contours, -1, (190, 221, 236), 3)

    blank = np.zeros(frame.shape, np.float64)
    blank[:, :, 0] = 44
    blank[:, :, 1] = 38
    blank[:, :, 2] = 114
    blcon = cv2.drawContours(blank, contours, -1, (190, 221, 236), 3)
    return blcon, img_con

def corner(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, 4, 5, 0.01)
    img_2 = img.copy()
    img_2[dst > 0.01 * dst.max()] = [205, 184, 239]

    return img_2

def edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.4 * high_thresh

    print(low_thresh)
    print(high_thresh)

    edge = cv2.Canny(gray, low_thresh, high_thresh, L2gradient= True)
    edge_c = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    edge_c[:, :, 0] = edge_c[:, :, 0]* (190 / 255)
    edge_c[:, :, 1] = edge_c[:, :, 1] * (221 / 255)
    edge_c[:, :, 2] = edge_c[:, :, 2] * (236 / 255)

    blank = np.zeros(frame.shape, np.float64)
    blank[:, :, 0] = 44
    blank[:, :, 1] = 38
    blank[:, :, 2] = 114

    img = (blank + edge_c)*((blank.sum())/(blank.sum()+ edge_c.sum()))
    #return img
    return img

n = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    keyPressed =cv2.waitKey(1)

    if keyPressed & 0xFF == ord('q'):
        break
    if keyPressed != -1 and keyPressed != 255 and keyPressed != cur_mode:
        cur_mode = keyPressed

    if cur_mode == ord('1'):
        cv2.imshow('project2',edge(frame).astype(np.uint8))
    elif cur_mode == ord('2'):
        contour_1, contour_2 = contour(frame)
        cv2.imshow('project2', contour_1.astype(np.uint8))
    elif cur_mode == ord('3'):
        contour_1, contour_2 = contour(frame)
        cv2.imshow('project2', contour_2.astype(np.uint8))
    elif cur_mode == ord('4'):
        cv2.imshow('project2', corner(frame))
    elif cur_mode == ord('5'):
        cv2.imshow('project2', sift(frame))
    else:
        cv2.imshow('project2',frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()