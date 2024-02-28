from modules.visual_memory import VisualMemory
from modules.localization import Localization
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import os
from scipy.spatial.transform import Rotation as R
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from scipy.spatial.transform import Rotation as R

class KeyBindings:
    FORWARD_KEY = "w"
    LEFT_KEY = "a"
    RIGHT_KEY = "d"
    FINISH_KEY = "f"
    MOVE_KEY = "m"
    RESET_VM_KEY = "k"
    NEXT_VM_KEY = "0"
    LOCALIZATION_KEY = "l"  # Added localization key

class NavigationPolicy:
    def __init__(self, config=None, registration_result=None, visual_memory=None):
        self.config = config
        self.registration_result = registration_result
        self.fit_threshold = config['navigation']['fit_threshold']
        self.forward_threshold = config['navigation']['forward_threshold']
        self.lateral_threshold = config['navigation']['lateral_threshold']
        self.yaw_threshold = config['navigation']['yaw_threshold']
        self.vm_path = config['paths']['VM_PATH']
        self.visual_memory = visual_memory or VisualMemory(config)
        self.localization = Localization(config)
        self.setup_fuzzy_control()

    def setup_fuzzy_control(self):
        # Define the universe of discourse for inputs and outputs using config parameters
        position_error = ctrl.Antecedent(np.arange(-self.config['fuzzy_navigation']['max_position_error'], 
                                                self.config['fuzzy_navigation']['max_position_error'], 
                                                self.config['fuzzy_navigation']['error_resolution']), 'position_error')
        orientation_error = ctrl.Antecedent(np.arange(-self.config['fuzzy_navigation']['max_orientation_error'], 
                                                    self.config['fuzzy_navigation']['max_orientation_error'], 
                                                    self.config['fuzzy_navigation']['error_resolution']), 'orientation_error')
        action = ctrl.Consequent(np.arange(0, 4, 1), 'action')  # 0: Stop, 1: Move Forward, 2: Turn Left, 3: Turn Right

        # Membership functions setup using config thresholds from 'fuzzy_navigation'
        position_error['Near'] = fuzz.trimf(position_error.universe, [self.config['fuzzy_navigation']['position_error']['Near']['start'], 
                                                                    self.config['fuzzy_navigation']['position_error']['Near']['peak'], 
                                                                    self.config['fuzzy_navigation']['position_error']['Near']['end']])
        position_error['Far'] = fuzz.trimf(position_error.universe, [self.config['fuzzy_navigation']['position_error']['Far']['start'], 
                                                                    self.config['fuzzy_navigation']['position_error']['Far']['peak'], 
                                                                    self.config['fuzzy_navigation']['position_error']['Far']['end']])
        orientation_error['Small'] = fuzz.trimf(orientation_error.universe, [self.config['fuzzy_navigation']['orientation_error']['Small']['start'], 
                                                                            self.config['fuzzy_navigation']['orientation_error']['Small']['peak'], 
                                                                            self.config['fuzzy_navigation']['orientation_error']['Small']['end']])
        orientation_error['Moderate'] = fuzz.trimf(orientation_error.universe, [self.config['fuzzy_navigation']['orientation_error']['Moderate']['start'], 
                                                                                self.config['fuzzy_navigation']['orientation_error']['Moderate']['peak'], 
                                                                                self.config['fuzzy_navigation']['orientation_error']['Moderate']['end']])
        orientation_error['Large'] = fuzz.trimf(orientation_error.universe, [self.config['fuzzy_navigation']['orientation_error']['Large']['start'], 
                                                                            self.config['fuzzy_navigation']['orientation_error']['Large']['peak'], 
                                                                            self.config['fuzzy_navigation']['orientation_error']['Large']['end']])
        
        action['Stop'] = fuzz.trimf(action.universe, [0, 0, 1])
        action['Move Forward'] = fuzz.trimf(action.universe, [1, 1, 2])
        action['Turn Left'] = fuzz.trimf(action.universe, [2, 2, 3])
        action['Turn Right'] = fuzz.trimf(action.universe, [3, 3, 3])

        # Fuzzy rules
        rule1 = ctrl.Rule(position_error['Near'] & orientation_error['Small'], action['Stop'])
        rule2 = ctrl.Rule(position_error['Far'] & orientation_error['Small'], action['Move Forward'])
        rule3 = ctrl.Rule(orientation_error['Large'], action['Turn Right'])
        rule4 = ctrl.Rule(orientation_error['Large'], action['Turn Left'])

        # Control system
        self.action_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
        self.action_decision = ctrl.ControlSystemSimulation(self.action_control)

    def print_success(self, message):
        green = "\033[92m"
        reset = "\033[0m"
        print(f"{green}Success: {message}{reset}")

    def print_warning(self, message):
        red = "\033[91m"
        reset = "\033[0m"
        print(f"{red}Warning: {message}{reset}")

    def determine_bot_action(self, registration_result):
        """
        Determine the action a bot should take based on the transformation matrix.
        Args:
        T (np.array): A 4x4 transformation matrix containing rotation and translation.
        Returns:
        str: The action the bot should take: 'Move Forward', 'Turn Right', 'Turn Left', or 'Stop'.
        """
        self.registration_result = registration_result

        if self.registration_result.fitness < self.fit_threshold:
            self.print_warning("Warning: Regitration failed. Fitness score: {}".format(self.registration_result.fitness))
        else:
            self.print_success("Registration successful. Fitness score: {}".format(self.registration_result.fitness))

        # Extract the translation vector and Euler angles
        print('Processing action')
        T = np.copy(self.registration_result.transformation)
        translation = T[0:3, 3]
        rotation_matrix = T[0:3, 0:3]
        euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
        
        print(f'Translation: {translation}')
        print(f'Angles (xyz): {euler_angles}')

        # Check translation for forward/backward movement
        if translation[0] < -self.forward_threshold:
            action_forward = 'Move Forward'
        else:
            action_forward = 'Stop'  # If the bot is close enough to the target

        # Check lateral translation and yaw angle for turning
        if translation[1] < -self.lateral_threshold or euler_angles[2] < -self.yaw_threshold:
            action_turn = 'Turn Right'
        elif translation[1] > self.lateral_threshold or euler_angles[2] > self.yaw_threshold:
            action_turn = 'Turn Left'
        else:
            action_turn = None  # No turn is needed if within thresholds

        # Combine actions: prioritize turning over moving forward
        if action_turn:
            return action_turn
        else:
            return action_forward
        
    def handle_keystroke(self, keystroke, vm_image_index, registration_result, current_color_image=None):
        if keystroke == ord(KeyBindings.MOVE_KEY):
            computed_action = 'Stop'

            computed_action = self.determine_bot_action(registration_result)

            if computed_action == 'Move Forward':
                action = HabitatSimActions.move_forward
            elif computed_action == 'Turn Right':
                action = HabitatSimActions.turn_right
            elif computed_action == 'Turn Left':
                action = HabitatSimActions.turn_left
            elif computed_action == 'Stop':
                vm_image_index = (vm_image_index + 1) % len(os.listdir(self.vm_path + "color/"))
                self.visual_memory.display_visual_memory( vm_image_index)
                return vm_image_index, None  # No action to execute

        elif keystroke == ord(KeyBindings.FORWARD_KEY):
            action = HabitatSimActions.move_forward

        elif keystroke == ord(KeyBindings.LEFT_KEY):
            action = HabitatSimActions.turn_left

        elif keystroke == ord(KeyBindings.RIGHT_KEY):
            action = HabitatSimActions.turn_right

        elif keystroke == ord(KeyBindings.FINISH_KEY):
            print("Finishing the episode.")
            return vm_image_index, "finish"  # Signal to finish the episode

        elif keystroke == ord(KeyBindings.RESET_VM_KEY):
            vm_image_index = 0
            self.visual_memory.display_visual_memory(vm_image_index)
            return vm_image_index, None  # No action to execute

        elif keystroke == ord(KeyBindings.NEXT_VM_KEY):
            vm_image_index = (vm_image_index + 1) % len(os.listdir(self.vm_path + "color/"))
            self.visual_memory.display_visual_memory(vm_image_index)
            return vm_image_index, None  # No action to execute
        
        elif keystroke == ord(KeyBindings.LOCALIZATION_KEY):
            if current_color_image is not None:
                # Call the localization method with the current color image
                best_match_name, best_distance_name, max_matches, min_avg_distance = self.localization.localization_in_visual_memory(current_color_image)
                print(f"Best match image: {best_match_name} | Number of Matches: {max_matches}")
                print(f"Best distance image: {best_distance_name} | Average Distance: {min_avg_distance:.2f}")
            else:
                print("No current image provided for localization.")
            return vm_image_index, None  # No action to execute after localization

        else:
            return vm_image_index, None  # No action for unrecognized keystrokes

        # For actions that involve moving the agent
        return vm_image_index, action
    
    def fuzzy_bot_action(self, registration_result):
        """
        Determine the action a bot should take based on the transformation matrix.
        Args:
        T (np.array): A 4x4 transformation matrix containing rotation and translation.
        Returns:
        str: The action the bot should take: 'Move Forward', 'Turn Right', 'Turn Left', or 'Stop'.
        """
        self.registration_result = registration_result

        if self.registration_result.fitness < self.fit_threshold:
            self.print_warning("Warning: Regitration failed. Fitness score: {}".format(self.registration_result.fitness))
        else:
            self.print_success("Registration successful. Fitness score: {}".format(self.registration_result.fitness))

        # Extract the translation vector and Euler angles
        print('Processing action')
        T = np.copy(self.registration_result.transformation)
        translation = T[0:3, 3]
        rotation_matrix = T[0:3, 0:3]
        euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
        
        print(f'Translation: {translation}')
        print(f'Angles (xyz): {euler_angles}')

        # Convert translation and rotation to position and orientation errors
        position_error_value = np.linalg.norm(translation)  # Simple distance for position error
        orientation_error_value = np.abs(euler_angles[2])  # Absolute yaw angle for orientation error

        # Fuzzy inference
        self.action_decision.input['position_error'] = position_error_value
        self.action_decision.input['orientation_error'] = orientation_error_value
        self.action_decision.compute()

        # Mapping the crisp action output to action commands
        action_value = self.action_decision.output['action']
        if action_value <= 0.5:
            return 'Stop'
        elif action_value <= 1.5:
            return 'Move Forward'
        elif action_value <= 2.5:
            return 'Turn Left'
        else:
            return 'Turn Right'
