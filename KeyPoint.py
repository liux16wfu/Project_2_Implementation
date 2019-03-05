import cv2
import numpy as np

cur_mode = None
cap = cv2.VideoCapture(0)
sift = cv2.xfeatures2d.SIFT_create(nfeatures = 50, contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6)
bfmatcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

def siftMatch(img, preImg):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(preImg, cv2.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(gray,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)


    matches = bfmatcher.match(des, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img, kp, preImg, kp2, matches, None)

    return img3


def cornerMatch(img, prevImg):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)

    corners = cv2.cornerHarris(gray, 4, 5, 0.02)
    corners2 = cv2.cornerHarris(gray2, 4, 5, 0.02)

    kpsCorners = np.argwhere(corners > 0.01 * corners.max())
    kpsCorners = [cv2.KeyPoint(pt[1], pt[0], 3) for pt in kpsCorners]

    kpsCorners2 = np.argwhere(corners2 > 0.01*corners2.max())
    kpsCorners2 = [cv2.KeyPoint(pt[1], pt[0], 3) for pt in kpsCorners2]

    kpsCorners, dscCorners = sift.compute(gray, kpsCorners)
    kpsCorners2, dscCorners2 = sift.compute(gray2, kpsCorners2)

    matchesCorner = bfmatcher.match(dscCorners,dscCorners2)
    matchesCorner = sorted(matchesCorner, key= lambda x:x.distance)
    img_2 = cv2.drawMatches(img, kpsCorners, prevImg, kpsCorners2, matchesCorner, None)

    return img_2


#ret, prev_img = cap.read()
n = 0
while(True):

    ret, frame = cap.read()
    img = frame

    prev_img = cv2.flip(img, 0)



    # if n == 20:
    #     prev_img = frame
    #     n = 0
    # else:
    #     n += 1



    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key != -1 and key != 255 and key != cur_mode:
        cur_mode = key

    if cur_mode == ord('1'):
        comparison = cornerMatch(img, prev_img)
        cv2.imshow('project2', comparison)
    elif cur_mode == ord('2'):
        siftmatch = siftMatch(img, prev_img)
        cv2.imshow('project2', siftmatch)
    else:
        cv2.imshow("project2", frame)



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()