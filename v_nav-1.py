# Modify the quantity of movement in:
# /home/rodriguez/Documents/GitHub/habitat/habitat-lab/habitat-lab/habitat/config/default_structured_configs.py 
# (again to avoid following the same visual path)
# Include the option for colored pc and color-icp
# Consider initial alligment techniques (including AI)
# Try and error evaluation (since the complete fails...)
# I observe that it diverges when the wall appears on the image
# PB: Fails until 24 using SG and ICP
# Re implement ORB, BRISK and AKAZE
# Colored points instead of lines
# Inifite loop; not limited to 500 iterations


import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import numpy as np
from omegaconf import OmegaConf
import yaml

import torch

from models.matching import Matching
from models.utils import frame2tensor

from modules.visual_memory import VisualMemory
from modules.navigation_policy import NavigationPolicy
from modules.registration import Registration
from modules.point_cloud import PointCloud
from modules.feature_matcher import FeatureMatcher
from modules.image_processor import ImageProcessor
from modules.localization import Localization
from sensors.range_sensor import SimulatedRangeSensor

def demo():
    # Define the device for computation at the start of the demo function

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    env = habitat.Env(config=habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml"))

    print("Environment creation successful")
    observations = env.reset()

    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))

    print("Agent stepping around inside environment.")
    
    with open('/home/rodriguez/Documents/GitHub/habitat/habitat-lab/examples/config.yaml') as f:
        config = yaml.safe_load(f)

    count_steps = 0
    vm_image_index = 0
    icp_result = 0 #

    ip = ImageProcessor(config=config)
    feature = FeatureMatcher(config=config, device=device)
    vm = VisualMemory(config=config)
    pc = PointCloud(config=config)
    registration = Registration(config=config)
    nav = NavigationPolicy(config=config)
    #loc = Localization(config=config)
    range_sensor = SimulatedRangeSensor(threshold_distance=0.3) # must be updated to the config

    current_color, current_depth = ip.display_current_images(observations["rgb"], observations["depth"])
    vm.display_visual_memory(vm_image_index)

    while not env.episode_over:
        target_color, target_depth = vm.load_vm_images(vm_image_index)

        vm.save_images(current_color, current_depth, count_steps)
        if not config["mode"]["visual_path"]:
            obstacle_left, obstacle_center, obstacle_right = range_sensor.process_depth_image(current_depth)
            print(f"Obstacle left: {obstacle_left}, center: {obstacle_center}, right: {obstacle_right}")
            
            # Feature matching part
            # Assume SuperGlue is used for simplicity; adapt as needed for other methods
            print("Using SuperGlue (only this one is currently implemented).")
            kp1, kp2 = feature.compute_matches(target_color, current_color, vm_image_index)
            print("Number of trusted matched points: ", len(kp1))
            #feature.save_matched_points(target_color, current_color, kp1, kp2, vm_image_index)

            # Point clouds part
            print(f"Initiating registration with distance threshold: {registration.distance_threshold}")
            pc1, pc2 = pc.get_3d_points(kp1, kp2, current_depth, target_depth, current_color, target_color, vm_image_index, count_steps)

            # Debugging stuff
            pc.merge_and_recolor_point_clouds() 

            # Registration part
            icp_result = registration.pc_registration(pc1, pc2)
            #print("ICP Registration result: ", icp_result.transformation)

            registration.save_colored_registration_result(count_steps) # Warning: always saving 

            computed_action = nav.determine_bot_action(icp_result) # if icp_result.fitness >= th_fit else 'Stop'
            print("Computed action: ", computed_action)

            #computed_fuzzy_action = nav.fuzzy_bot_action(icp_result) # if icp_result.fitness >= th_fit else 'Stop'
            #print("Computed fuzzy action: ", computed_fuzzy_action)

        print('Step: ', count_steps)
        keystroke = cv2.waitKey(0)
        vm_image_index, action = nav.handle_keystroke(keystroke, vm_image_index, icp_result, current_color)

        if action == "finish":
            break
        elif action:
            observations = env.step(action)
            # Update the current_color and current_depth after the action
            current_color, current_depth = ip.display_current_images(observations["rgb"], observations["depth"])
            count_steps += 1


    print("Episode finished after {} steps.".format(count_steps))

if __name__ == "__main__":
    demo()
