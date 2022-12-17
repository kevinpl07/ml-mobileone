import cv2
import numpy as np
import json
import os



def get_per_channel_sharpness(img):

    blue_gy, blue_gx = np.gradient(img[:,:,0])
    green_gy, green_gx = np.gradient(img[:,:,1])
    red_gy, red_gx = np.gradient(img[:,:,2])

    blue_gnorm = np.sqrt(blue_gx**2 + blue_gy**2)
    green_gnorm = np.sqrt(green_gx**2 + green_gy**2)
    red_gnorm = np.sqrt(red_gx**2 + red_gy**2)

    blue_sharpness = np.average(blue_gnorm)
    green_sharpness = np.average(green_gnorm)
    red_sharpness = np.average(red_gnorm)

    return blue_sharpness, green_sharpness, red_sharpness

vid_mat_img = cv2.imread("C:\\Users\\kbond\\OneDrive\\Desktop\\projects\\VideoMatte240K_JPEG_HD\\test\\fgr\\0000\\00000.jpg")

whamen_img = cv2.imread(".\\whamen_imgs\\image (4).png")

label_file = ".\\whamen_imgs\\labels.json"


vid_mat_img = cv2.resize(vid_mat_img, whamen_img.shape[:2])

with open(label_file, 'r', encoding="utf-8") as f:
    labels = json.load(f)

label_dict = labels["image (4).png"]
   
mask = np.zeros(whamen_img.shape[:2], dtype=np.uint8)
# Mask out foreground objects
for region_key, region_dict in label_dict["regions"].items():
    if region_dict['region_attributes']["label"] != "Background":
        polygon = region_dict['shape_attributes']
        x_points = np.asarray(polygon['all_points_x'])
        y_points = np.asarray(polygon['all_points_y'])
        points = np.asarray(list(zip(x_points, y_points)))
        cv2.fillPoly(mask, np.asarray([points], dtype=np.int32), 1)

for region_key, region_dict in label_dict["regions"].items():
    if region_dict['region_attributes']["label"] == "Background":
        polygon = region_dict['shape_attributes']
        x_points = np.asarray(polygon['all_points_x'])
        y_points = np.asarray(polygon['all_points_y'])
        points = np.asarray(list(zip(x_points, y_points)))
        cv2.fillPoly(mask, np.asarray([points], dtype=np.int32), 0)


wh_blue_sharpness, wh_green_sharpness, wh_red_sharpness = get_per_channel_sharpness(whamen_img)
print(get_per_channel_sharpness(whamen_img))

test_blue_sharpness, test_green_sharpness, test_red_sharpness = get_per_channel_sharpness(vid_mat_img)
print(get_per_channel_sharpness(vid_mat_img))


import pdb; pdb.set_trace()