import numpy as np
import cv2
import torch
import os
from models.matching import Matching
from models.utils import frame2tensor

class Localization:
    def __init__(self, config, device='cpu', c_img=None, t_img=None):
        self.config = config
        self.device = device
        self.c_img = c_img
        self.t_img = t_img

    def compute_confident_matches_and_distances(self):
        """
        Computes the number of confident matches and the distances of the matched points
        between the current and target images using SuperGlue.

        :param current_image: The current image as a numpy array.
        :param target_image: The target image as a numpy array.
        :return: A tuple containing the number of confident matches and an array of distances.
        """

        # Convert images to grayscale
        g_image1 = cv2.cvtColor(self.c_img, cv2.COLOR_BGR2GRAY)
        g_image2 = cv2.cvtColor(self.t_img, cv2.COLOR_BGR2GRAY)

        # Convert images to tensor format
        frame_tensor1 = frame2tensor(g_image1, self.device)
        frame_tensor2 = frame2tensor(g_image2, self.device)

        # Initialize the matching model with SuperGlue
        matching = Matching({'superpoint': {}, 'superglue': {'weights': 'indoor'}}).to(self.device).eval()

        # Perform matching
        with torch.no_grad():  # No need to track gradients
            pred = matching({'image0': frame_tensor1, 'image1': frame_tensor2})

        # Extract keypoints and matches
        kpts0 = pred['keypoints0'][0].cpu().detach().numpy()
        kpts1 = pred['keypoints1'][0].cpu().detach().numpy()
        matches = pred['matches0'][0].cpu().detach().numpy()
        confidence = pred['matching_scores0'][0].cpu().detach().numpy()

        # Filter matches based on confidence
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        conf = confidence[valid]

        # Select high-confidence points
        threshold = self.config['thresholds']['confidence']  # Adjust this threshold as needed
        kp1, kp2 = self.select_high_confidence_points_with_superglue(mkpts0, mkpts1, conf, threshold)

        # Compute distances between matched keypoints
        distances = np.linalg.norm(kp1 - kp2, axis=1) if len(kp1) > 0 else np.array([])

        # Determine the number of confident matches
        num_confident_matches = len(kp1)

        return num_confident_matches, distances

    def select_high_confidence_points_with_superglue(self, mkp1, mkp2, confidences, threshold):
        kp1 = []
        kp2 = []

        for i, confidence in enumerate(confidences):
            if confidence > threshold:
                kp1.append(mkp1[i])
                kp2.append(mkp2[i])

        return np.asarray(kp1), np.asarray(kp2)


    def is_localization_reliable(self, num_confident_matches, distances):
        """
        Determines if the localization is reliable based on the number of confident matches
        and the distances of the matched points.

        :param num_confident_matches: The number of confident matches between the images.
        :param distances: An array of distances between the matched points.
        :return: A boolean indicating whether the localization is reliable.
        """
        # Retrieve the confidence threshold from the configuration
        match_threshold = self.config["localization"]["match_threshold"]
        distance_threshold = self.config["localization"]["distance_threshold"]

        # Check if the number of confident matches is above the threshold
        if num_confident_matches < match_threshold:
            return False

        # Calculate the average distance of the matched points
        if distances.size > 0:
            avg_distance = np.mean(distances)
        else:
            # If there are no distances, consider localization unreliable
            return False

        # Check if the average distance is below the threshold
        if avg_distance > distance_threshold:
            return False

        # If both conditions are satisfied, consider the localization reliable
        return True
        
    def localization_in_visual_memory(self, c_img):
        """
        Compares the current image with all images in the visual memory to find the best match.
        
        :param c_img: The current image as a numpy array.
        :return: A tuple containing the names of the image with the most matches and the image with the lowest average distance, along with their respective scores.
        """
        self.c_img = c_img  # Update the current image
        vm_path = self.config['paths']['VM_PATH'] + 'color/'
        images = os.listdir(vm_path)

        max_matches = 0
        min_avg_distance = float('inf')
        best_match_index = -1
        best_distance_index = -1

        for idx, img_name in enumerate(images):
            img_path = os.path.join(vm_path, img_name)
            self.t_img = cv2.imread(img_path)  # Update the target image

            # Check if the image was loaded successfully
            if self.t_img is None or self.t_img.size == 0:
                print(f"Warning: Could not load image at {img_path}. Skipping...")
                continue

            # Compute the number of confident matches and distances
            num_confident_matches, distances = self.compute_confident_matches_and_distances()

            # Update the best match based on the number of matches
            if num_confident_matches > max_matches:
                max_matches = num_confident_matches
                best_match_index = idx

            # Update the best match based on the lowest average distance
            if distances.size > 0:
                avg_distance = np.mean(distances)
                if avg_distance < min_avg_distance:
                    min_avg_distance = avg_distance
                    best_distance_index = idx

        best_match_name = images[best_match_index] if best_match_index != -1 else "None"
        best_distance_name = images[best_distance_index] if best_distance_index != -1 else "None"

        return best_match_name, best_distance_name, max_matches, min_avg_distance
