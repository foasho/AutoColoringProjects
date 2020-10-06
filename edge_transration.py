import cv2
import numpy as np
import os

#線画変換
def edge_detect(cv2img):
    gray = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)
    cv2.merge((gray, gray, gray), cv2img)
    kernel = np.ones((4, 4), np.uint8)
    dilation = cv2.dilate(cv2img, kernel, iterations=1)
    diff = cv2.subtract(dilation, cv2img)
    negaposi = 255 - diff
    return negaposi

if __name__=='__main__':
    target_dir = "./images/color/"
    save_dir_edge = "./images/edge/"
    if not os.path.exists(save_dir_edge):
        os.makedirs(save_dir_edge)
    image_names = os.listdir(target_dir)
    for image_name in image_names:
        read_path = target_dir + image_name
        try:
            color_image = cv2.imread(read_path)
            edge_image = edge_detect(color_image)
            cv2.imwrite(save_dir_edge + image_name + ".jpg", edge_image)
        except Exception as e:
            print(read_path)
            print("error")
            print(e)