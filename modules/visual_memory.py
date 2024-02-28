import os
import cv2

class VisualMemory:
    def __init__(self, config, current_color=None, target_color=None, current_depth=None, target_depth=None, id=0):
        self.config = config
        self.current_color = current_color
        self.target_color = target_color
        self.current_depth = current_depth
        self.target_depth = target_depth
        self.id = id
        self.logs_dir = self.config['paths']['LOGS_DIR']
        self.save_current_images = self.config['logs']['current_imgs']

    def display_visual_memory(self, current_image_index):
        image_folder = self.config['paths']['VM_PATH']
        color_image_folder = os.path.join(image_folder, "color/")
        depth_image_folder = os.path.join(image_folder, "depth/")

        color_image_files = sorted(os.listdir(color_image_folder))
        depth_image_files = sorted(os.listdir(depth_image_folder))

        if current_image_index < len(color_image_files) and current_image_index < len(depth_image_files):
            color_image_path = os.path.join(color_image_folder, color_image_files[current_image_index])
            depth_image_path = os.path.join(depth_image_folder, depth_image_files[current_image_index])

            color_image = cv2.imread(color_image_path)
            depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

            cv2.namedWindow("Target Color", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("Target Depth", cv2.WINDOW_AUTOSIZE)

            # Move the windows to the specified initial positions
            cv2.moveWindow("Target Color", 500, 100)
            cv2.moveWindow("Target Depth", 500, 500)

            cv2.imshow("Target Color", color_image)
            cv2.imshow("Target Depth", depth_image)

    def load_vm_images(self, current_image_index):
        vm_path = self.config['paths']['VM_PATH']
        vm_path_color = vm_path + "color/"
        color_files = sorted(os.listdir(vm_path_color))
        if current_image_index < len(color_files):
            image_path = os.path.join(vm_path_color, color_files[current_image_index])
            color_vm_image = cv2.imread(image_path)

        vm_path_depth = vm_path + "depth/"
        depth_files = sorted(os.listdir(vm_path_depth))
        if current_image_index < len(depth_files):
            image_path = os.path.join(vm_path_depth, depth_files[current_image_index])
            depth_vm_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        return color_vm_image, depth_vm_image
    
    def save_images(self, rgb_image, depth_image, step_count):
        if self.save_current_images:
            color_path = self.logs_dir+f"all_images/color/{step_count:04d}.png"
            depth_path = self.logs_dir+f"all_images/depth/{step_count:04d}.png"
            cv2.imwrite(color_path, rgb_image)
            cv2.imwrite(depth_path, depth_image)