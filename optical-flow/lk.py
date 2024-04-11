import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import cv2

SLICE = 40
IMAGE_PATH = r"\ax_stripped.nii.gz"


def draw_circles(image, features):
    for pt in features:
        x, y = pt.ravel()
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), radius=1, color=(100, 0, 0), thickness=-1)

    cv2.imshow("Image with Features", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_arrows(image1, image2, flow, features):
    for i in range(len(flow)):
        x, y = features[i].ravel()
        x, y = int(x), int(y)
        dx, dy = flow[i].ravel()
        cv2.arrowedLine(image1, (x, y), (int(dx), int(dy)), (100, 0, 0), thickness=1)

    cv2.imshow('Flow Visualization', image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = sitk.ReadImage(IMAGE_PATH)
    image_arr = sitk.GetArrayFromImage(image)
    image_arr_8bit = cv2.normalize(image_arr, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    slice1 = image_arr_8bit[SLICE]
    slice2 = image_arr_8bit[SLICE+1]

    params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    features1 = cv2.goodFeaturesToTrack(slice1, maxCorners=150, qualityLevel=0.01, minDistance=10)
    features2 = cv2.goodFeaturesToTrack(slice2, maxCorners=150, qualityLevel=0.01, minDistance=10)
    flow, status, err = cv2.calcOpticalFlowPyrLK(prevImg=slice1, 
                                    nextImg=slice2, 
                                    prevPts=features1, 
                                    nextPts=None, 
                                    **params)
    # draw_circles(slice1, features1)
    # draw_circles(slice2, features2)
    draw_arrows(slice1, slice2, flow, features1)

