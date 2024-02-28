import os
import cv2
import numpy as np
import torch

from models.matching import Matching
from models.utils import frame2tensor

class FeatureMatcher:
    def __init__(self, config, device='cpu', c_img=None, t_img=None):
        self.config = config
        self.device = device
        self.c_img = c_img
        self.t_img = t_img
        self.filtered_matched_points = None
        self.vm_id = 0

    def set_target(self, t_img):
        self.t_img = t_img

    def set_current(self, c_img):
        self.c_img = c_img

    def compute_matches(self, c_img, t_img, vm_id, threshold=0.5):
        self.vm_id = vm_id
        if self.config['feature_matching']['descriptor'] == 'SuperGlue':
            threshold = self.config['thresholds']['confidence']
            kp1, kp2 = self.filtered_matched_points_with_superglue(c_img, t_img, threshold)
        if self.config['feature_matching']['descriptor'] == 'AKAZE':
            kp1, kp2 = self.matched_points_with_akaze(c_img, t_img, threshold)
        if self.config['feature_matching']['descriptor'] == 'ORB':
            kp1, kp2 = self.matched_points_with_orb(c_img, t_img, threshold)
        if self.config['feature_matching']['descriptor'] == 'BRISK':
            kp1, kp2 = self.matched_points_with_brisk(c_img, t_img, threshold)
        self.save_matched_points(c_img, t_img, kp1, kp2)
            
        return kp1, kp2

    def matched_points_with_akaze(self, c_img, t_img, threshold=0.5):
        # Initialize AKAZE detector
        akaze = cv2.AKAZE_create()
        # Find the keypoints and descriptors with AKAZE
        kp1, des1 = akaze.detectAndCompute(c_img, None)
        kp2, des2 = akaze.detectAndCompute(t_img, None)
        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)
        
        # Extract location of good matches
        points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

        for i, match in enumerate(good_matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        return points1, points2

    def matched_points_with_orb(self, c_img, t_img, threshold=0.5):
        # Initialize ORB detector
        orb = cv2.ORB_create()
        # Find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(c_img, None)
        kp2, des2 = orb.detectAndCompute(t_img, None)
        
        # BFMatcher with default params and crossCheck=True for better matches
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        return points1, points2

    def matched_points_with_brisk(self, c_img, t_img, threshold=0.5):
        # Initialize BRISK detector
        brisk = cv2.BRISK_create()
        # Find the keypoints and descriptors with BRISK
        kp1, des1 = brisk.detectAndCompute(c_img, None)
        kp2, des2 = brisk.detectAndCompute(t_img, None)
        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)
        
        # Extract location of good matches
        points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

        for i, match in enumerate(good_matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        return points1, points2

    '''
    def save_matched_points(self, c_img, t_img, kp1, kp2):
        if self.config['feature_matching']['descriptor'] == 'SuperGlue' and self.config['logs']['matched_points'] and len(kp1) > 0:
            matched_image = self.save_matched_points_with_superglue(c_img, t_img, kp1, kp2)
            match_img_path = os.path.join(self.config['paths']['LOGS_DIR'], f"matched_points/match_{self.vm_id:04d}_{len(kp1)}.png")
            cv2.imwrite(match_img_path, matched_image)       
        # Other features are not supported yet
    '''
    #############
    def save_matched_points(self, c_img, t_img, kp1, kp2):
        # Check if matched points logging is enabled in the configuration
        if self.config['logs']['matched_points'] and len(kp1) > 0:
            descriptor = self.config['feature_matching']['descriptor']
            matched_image = self.visualize_matched_points_by_color(c_img, t_img, kp1, kp2, descriptor)
            match_img_path = os.path.join(self.config['paths']['LOGS_DIR'], f"matched_points/{descriptor}_match_{self.vm_id:04d}_{len(kp1)}.png")
            cv2.imwrite(match_img_path, matched_image)

    def visualize_matched_points(self, c_img, t_img, kp1, kp2, descriptor):
        """ Draw lines connecting matched keypoints between image1 and image2 for any descriptor. """
        if len(c_img.shape) == 2:
            c_img = cv2.cvtColor(c_img, cv2.COLOR_GRAY2BGR)
        if len(t_img.shape) == 2:
            t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2BGR)

        h1, w1 = c_img.shape[:2]
        h2, w2 = t_img.shape[:2]
        height = max(h1, h2)
        width = w1 + w2
        matched_image = np.zeros((height, width, 3), dtype=np.uint8)
        matched_image[:h1, :w1] = c_img
        matched_image[:h2, w1:w1+w2] = t_img

        for pt1, pt2 in zip(kp1, kp2):
            pt1 = (int(round(pt1[0])), int(round(pt1[1])))
            pt2 = (int(round(pt2[0] + w1)), int(round(pt2[1])))

            cv2.circle(matched_image, pt1, 3, (0, 255, 0), -1)
            cv2.circle(matched_image, pt2, 3, (0, 255, 0), -1)
            cv2.line(matched_image, pt1, pt2, (255, 0, 0), 1)

        # Resize for visualization if needed
        matched_image = cv2.resize(matched_image, (width, height))

        return matched_image

    def visualize_matched_points_by_color(self, c_img, t_img, kp1, kp2, descriptor):
        """ Visualize matched keypoints in different colors without connecting lines. """
        if len(c_img.shape) == 2:
            c_img = cv2.cvtColor(c_img, cv2.COLOR_GRAY2BGR)
        if len(t_img.shape) == 2:
            t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2BGR)

        h1, w1 = c_img.shape[:2]
        h2, w2 = t_img.shape[:2]
        height = max(h1, h2)
        width = w1 + w2
        matched_image = np.zeros((height, width, 3), dtype=np.uint8)
        matched_image[:h1, :w1] = c_img
        matched_image[:h2, w1:w1+w2] = t_img

        num_matches = len(kp1)
        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_matches)]

        for idx, (pt1, pt2) in enumerate(zip(kp1, kp2)):
            color = colors[idx]
            pt1 = (int(round(pt1[0])), int(round(pt1[1])))
            pt2 = (int(round(pt2[0] + w1)), int(round(pt2[1])))

            cv2.circle(matched_image, pt1, 3, color, -1)
            cv2.circle(matched_image, pt2, 3, color, -1)

        # Resize for visualization if needed
        matched_image = cv2.resize(matched_image, (width, height))

        return matched_image
    #############

    def save_matched_points_with_superglue(self, c_img, t_img, kp1, kp2):
        """ Draw lines connecting matched keypoints between image1 and image2. """
        if len(c_img.shape) == 2:
            c_img = cv2.cvtColor(c_img, cv2.COLOR_GRAY2BGR)
        if len(t_img.shape) == 2:
            t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2BGR)

        h1, w1 = c_img.shape[:2]
        h2, w2 = t_img.shape[:2]
        height = max(h1, h2)
        width = w1 + w2
        matched_image = np.zeros((height, width, 3), dtype=np.uint8)
        matched_image[:h1, :w1] = c_img
        matched_image[:h2, w1:w1+w2] = t_img

        for pt1, pt2 in zip(kp1, kp2):
            pt1 = (int(round(pt1[0])), int(round(pt1[1])))
            pt2 = (int(round(pt2[0] + w1)), int(round(pt2[1])))

            cv2.circle(matched_image, pt1, 3, (0, 255, 0), -1)
            cv2.circle(matched_image, pt2, 3, (0, 255, 0), -1)
            cv2.line(matched_image, pt1,pt2, (255, 0, 0), 1)

        # Resize for visualization
        matched_image = cv2.resize(matched_image, (width , height))

        return matched_image

    def match_with_superglue(self, c_img, t_img):
        self.c_img = c_img
        self.t_img = t_img

        g_image1 = cv2.cvtColor(self.c_img, cv2.COLOR_BGR2GRAY)
        g_image2 = cv2.cvtColor(self.t_img, cv2.COLOR_BGR2GRAY)
        frame_tensor1 = frame2tensor(g_image1, self.device)
        frame_tensor2 = frame2tensor(g_image2, self.device)
        
        matching = Matching({'superpoint': {}, 'superglue': {'weights': 'indoor'}}).to(self.device).eval()
        with torch.no_grad():  # No need to track gradients
            pred = matching({'image0': frame_tensor1, 'image1': frame_tensor2})
        
        # Detach tensors before converting to NumPy arrays
        kpts0 = pred['keypoints0'][0].cpu().detach().numpy()
        kpts1 = pred['keypoints1'][0].cpu().detach().numpy()
        matches = pred['matches0'][0].cpu().detach().numpy()
        confidence = pred['matching_scores0'][0].cpu().detach().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        conf = confidence[valid]

        return mkpts0, mkpts1, conf

    def select_high_confidence_points_with_superglue(self, mkp1, mkp2, confidences, threshold):
        kp1 = []
        kp2 = []

        for i, confidence in enumerate(confidences):
            if confidence > threshold:
                kp1.append(mkp1[i])
                kp2.append(mkp2[i])

        return np.asarray(kp1), np.asarray(kp2)
    
    def filtered_matched_points_with_superglue(self, c_img, t_img, threshold):
        mkpts0, mkpts1, conf = self.match_with_superglue(c_img, t_img)
        kp1, kp2 = self.select_high_confidence_points_with_superglue(mkpts0, mkpts1, conf, threshold)
        self.kp1 = kp1
        self.kp2 = kp2
        return self.kp1, self.kp2
