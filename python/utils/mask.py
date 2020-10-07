import cv2
import numpy as np

# remove trash contours except the main one (if existed)
# input image is one-channel grayscale
def remove_trash(img):
    image = np.zeros((img.shape[0]+2, img.shape[1]+2), dtype=np.uint8)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[1:image.shape[0]-1, 1:image.shape[1]-1] = img
    _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if (area > largest_area):
            largest_area = area
            largest_contour = i
    M = cv2.moments(largest_contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    output_size = img.shape[0], img.shape[1]
    output_img = np.zeros(output_size, dtype=np.uint8)
    cv2.floodFill(output_img, image, (cx, cy), 255)
    cv2.circle(output_img, (cx, cy), 5, color= 255, thickness=3)
    return output_img



# def contour_to_mask(img_path):
#     im_in = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)
#     im_floodfill = im_th.copy()
#     h, w = im_th.shape[:2]
#     mask = np.zeros((h+2, w+2), np.uint8)
#
#     cv2.floodFill(im_floodfill, mask, (0,0), 255)
#
#     im_floodfill_inv = cv2.bitwise_not(im_floodfill)
#
#     _, contours, _, = cv2.findContours(im_floodfill_inv.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def fill_holes(gray_img):
    th, im_th = cv2.threshold(gray_img, 220, 255, cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0,0), 255)

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    return im_floodfill_inv

def remove_trash_2(binary_img):
    print('shape binary img', binary_img.shape)
    work_size = binary_img.shape[0]+2, binary_img.shape[1]+2
    work_image = np.zeros(work_size, dtype=np.uint8)
    work_image[1:work_image.shape[0]-1, 1:work_image.shape[1]-1] = binary_img
    contours = cv2.findContours(work_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    print(contours)
    largest_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if (area > largest_area):
            largest_area = area
            largest_contour = i
    M = cv2.moments(largest_contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    output_size = binary_img.shape[:2]
    output_img = np.zeros(output_size, dtype=np.uint8)
    work_image = 255 - work_image
    cv2.floodFill(output_img, work_image, (cx, cy), 255)
    return output_img
