import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, config, c_color=None, t_color=None, c_depth=None, t_depth=None):
        self.config = config
        self.c_color = c_color
        self.t_color = t_color
        self.c_depth = c_depth
        self.t_depth = t_depth
    
    def transform_rgb_bgr(self, color_image):
        return color_image[:, :, [2, 1, 0]]

    def transform_depth(self, image):
        depth_image = (1.0 - (image / np.max(image))) * 255.0
        depth_image = depth_image.astype(np.uint8)
        return depth_image
    
    def display_current_images(self, color_image, depth_image):
        self.c_color = self.transform_rgb_bgr(color_image)
        self.c_depth = self.transform_depth(depth_image)
        cv2.namedWindow("Current Color", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Current Depth", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("Current Color", 100, 100)
        cv2.moveWindow("Current Depth", 100, 500)
        cv2.imshow("Current Color", self.c_color)
        cv2.imshow("Current Depth", self.c_depth)
        return self.c_color, self.c_depth
