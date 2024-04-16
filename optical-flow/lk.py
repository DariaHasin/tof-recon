import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import cv2

################################## OpticalFlowPyrLK ##################################
def drawArrows(image, flow, features):
    image_bright = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
    image_color = cv2.cvtColor(image_bright, cv2.COLOR_GRAY2BGR)

    for i in range(len(flow)):
        x, y = features[i].ravel()
        x, y = int(x), int(y)
        dx, dy = flow[i].ravel()
        cv2.circle(image_color, (x, y), radius=1, color=(0, 255, 0), thickness=-1)
        cv2.arrowedLine(image_color, (x, y), (int(dx), int(dy)), (0, 0, 255), thickness=1)
    return image_color


def applyOpticalFlowPyrLKOneDirection(prevSlice, nextSlice):
    params = dict(winSize=(30, 30), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    features = cv2.goodFeaturesToTrack(prevSlice, maxCorners=100, qualityLevel=0.01, minDistance=10)#, useHarrisDetector=True)
    flow, status, err = cv2.calcOpticalFlowPyrLK(prevImg=prevSlice, 
                                    nextImg=nextSlice, 
                                    prevPts=features, 
                                    nextPts=None, 
                                    **params)
    image_color = drawArrows(prevSlice, flow, features)
    return image_color


def showOpticalFlowPyrLKTwoDirections(slice1, slice2):
    image1_color = applyOpticalFlowPyrLKOneDirection(slice1, slice2)
    image2_color = applyOpticalFlowPyrLKOneDirection(slice2, slice1)
    horizontal_concat = cv2.hconcat([image1_color, image2_color])
    cv2.imshow('Flow Visualization', horizontal_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

################################ OpticalFlowFarneback ################################
def normalizeFlow(flow):
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = cv2.normalize(magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX)
    hue = angle * 180 / np.pi / 2
    hsv = cv2.cvtColor(magnitude[..., None], cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = hue
    hsv[..., 1] = 255
    hsv[..., 2] = magnitude
    flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    flow_img = flow_img.astype(np.uint8)
    return flow_img


def showOpticalFlowFarneback(slice1, slice2):
    flow = cv2.calcOpticalFlowFarneback(prev=slice1, next=slice2, 
                                        flow=None, pyr_scale=0.5, 
                                        levels=3, winsize=15, 
                                        iterations=15, poly_n=7, 
                                        poly_sigma=1.5, flags=0)
    flow_img = normalizeFlow(flow)
    flow_img_color = cv2.applyColorMap(flow_img, cv2.COLORMAP_TURBO)

    image_bright = cv2.convertScaleAbs(slice2, alpha=3, beta=20)
    image_color = cv2.cvtColor(image_bright, cv2.COLOR_GRAY2BGR)

    alpha = 0.4
    overlay = cv2.addWeighted(flow_img_color, alpha, image_color, 1-alpha, 0)
    cv2.imshow('Flow Visualization', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
######################################################################################

if __name__ == '__main__':
    image_path = r"\\10.101.119.14\Dafna\Daria\scans_stripped\Daria\ax_stripped.nii.gz"
    image = sitk.ReadImage(image_path)
    image_arr = sitk.GetArrayFromImage(image)
    image_arr_8bit = cv2.normalize(image_arr, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    shape = image_arr_8bit.shape
    for i in range(30, shape[0]-1):
        slice1 = image_arr_8bit[i]
        slice2 = image_arr_8bit[i+1]
        # showOpticalFlowPyrLKTwoDirections(slice1, slice2)
        showOpticalFlowFarneback(slice1, slice2)




