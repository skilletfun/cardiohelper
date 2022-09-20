import base64
import numpy as np

import cv2 as cv
import pytesseract
from scipy import ndimage, io

from predict import predictor


def sharpen(img):
    kernel = np.array([[0, -1, 0],
                        [-1, 5.5, -1],
                        [0, -1, 0]], np.float32)
    img = cv.filter2D(img, -1, kernel)
    return img

# Helper function to increase contrast of an image
def increase_contrast(img):
    lab_img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    l, a, b = cv.split(lab_img)
    clahe = cv.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img = cv.merge((cl, a, b))
    img = cv.cvtColor(img, cv.COLOR_LAB2RGB)
    return img

# Another helper function to crop and remove the borders
def crop_image_v2(image, tolerance=0):
    mask = image > tolerance
    image = image[np.ix_(mask.any(1), mask.any(0))]
    return image

# Helper function to distinguish different ECG signals on specific image
def separate_components(image):
    ret, labels = cv.connectedComponents(image, connectivity=8)

    # mapping component labels to hue value
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_image = cv.merge([label_hue, blank_ch, blank_ch])
    labeled_image = cv.cvtColor(labeled_image, cv.COLOR_HSV2BGR)

    # set background label to white
    labeled_image[label_hue == 0] = 255
    return labeled_image


# Helper function to detect characters
def ocr(image):
    text = pytesseract.image_to_string(image, lang='eng')
    return text

def convert_image_to_mat(image64):
    # select image
    decode = base64.b64decode(image64)
    nparr = np.fromstring(decode, np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)

    # use blurring and thresholding to transform the image into a binary one
    blurred_image = cv.GaussianBlur(image, (3, 3), 0)
    blurred_image = cv.medianBlur(blurred_image, 3)
    binary_image = cv.adaptiveThreshold(blurred_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 101, 50)

    # labeling regions of interest
    structure = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], np.uint8)
    labeled_image, nb = ndimage.label(binary_image, structure=structure)

    curve_indices, curve_lengths, curve_widths, curve_lower_bound, curve_upper_bound = [], [], [], [], []

    for i in range(1, np.amax(labeled_image) + 1):
        sl = ndimage.find_objects(labeled_image == i)
        img = binary_image[sl[0]]
        if img.shape[1] > 200 and img.shape[0] > 80:
            curve_indices.append(i)
            curve_widths.append(img.shape[0])
            curve_lengths.append(img.shape[1])
            curve_lower_bound.append(sl[0][0].stop)
            curve_upper_bound.append(sl[0][0].start)
        else:
            continue

    # for recording the baselines of the curves
    baselines = []
    for i in range(len(curve_indices)):
        sl = ndimage.find_objects(labeled_image == curve_indices[i])
        img = binary_image[sl[0]]
        maxx, line_num = 0, 0
        for k in range(img.shape[0]):
            cnt = 0
            for n in range(img.shape[1]):
                if img[k][n] == 255:
                    cnt += 1
            if cnt > maxx:
                maxx = cnt
                line_num = k
        baselines.append(curve_widths[curve_lengths.index(max(curve_lengths))] - line_num)

    sl = ndimage.find_objects(labeled_image == curve_indices[curve_lengths.index(max(curve_lengths))])
    final_img = binary_image[sl[0]]

    xs, ys = [], []
    curve = final_img
    length = curve.shape[1]
    width = curve.shape[0]
    for j in range(length):
        for k in range(width - 1, -1, -1):
            if curve[k][j] == 255:
                xs.append(j)
                ys.append(width - k - baselines[curve_lengths.index(max(curve_lengths))])
                break
            else:
                continue

    io.savemat('predict/matlab_file.mat', {'val': ys})
    
def main(image64):
    convert_image_to_mat(image64)
    return predictor.start_predict()
