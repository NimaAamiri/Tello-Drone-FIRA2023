import os
import threading
import time

import cv2
import numpy as np
import pygame
import torch
from djitellopy import Tello
from pupil_apriltags import Detector
from pygame.locals import *
from pygame.math import clamp


# inisialisation
class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            -   b:      Takeoff
            - Shift:    Land
            - Space:    Emergency Shutdown
            - WASD:     Forward, backward, left and right
            - Q and E:  Counter clockwise and clockwise rotations
            - R and F:  Up and down
            - T:        Start/Stop tracking
            - C:        Select central pixel value as new color for tracking
            - #:        Switch controllable parameter
            - + and -:  Raise or Lower controllable parameter
    """

    def __init__(self):
        # Init pygame
        self.should_stop = None
        pygame.init()

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()
        self.tag_detector = Detector(families="tag36h11")

        # general config
        self.internalSpeed = 100
        self.FPS = 20
        self.hud_size = (960, 720)
        self.bcorners = None  # Initialize bcorners as None
        self.aparea = 0

        # apriltag frame
        self.t_camera_params = [921.170702, 919.018377, 459.904354, 351.238301]
        self.tagsize = 0.079
        self.desired_area = 8000
        self.yaw_angle_deg = 0
        self.pitch_angle_deg = 0
        self.roll_angle_deg = 0

        # Yolo Model
        self.model_name = 'best.pt'
        self.Yolomodel = torch.hub.load(f"{os.getcwd()}/yolov5", 'custom', source='local', path=self.model_name,
                                        force_reload=True)
        self.Yolomodel.eval()

        # config of controllable parameters
        self.controll_params = {
            'Speed': 25,
            'Color': 0,
        }
        self.controll_params_d = {
            'Speed': 10,
            'Color': 1,
        }
        self.controll_params_m = {
            'Speed': 100,
            'Color': 2,
        }

        # tracker config
        self.color_lower = {
            'blue': (100, 200, 50),
            'red': (0, 200, 100),
            'yellow': (20, 200, 130),
        }
        self.color_upper = {
            'blue': (140, 255, 255),
            'red': (20, 255, 255),
            'yellow': (40, 255, 255),
        }

        self.current_color = np.array(self.color_lower['blue']) + np.array(self.color_upper['blue'])
        for i in range(0, 3): self.current_color[i] = self.current_color[i] / 2
        self.crange = (10, 50, 50)

        # other params (no need to config)
        self.current_parameter = 0
        self.param_keys = list(self.controll_params.keys())
        self.color_keys = list(self.color_lower.keys())
        self.central_color = (0, 0, 0)
        self.midx = int(self.hud_size[0] / 2)
        self.midy = int(self.hud_size[1] / 2 + self.hud_size[1] / 4)
        self.xoffset = 0
        self.yoffset = 0
        self.target_radius = 120
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.send_rc_control = False
        self.isTracking = False
        self.isGoingThruGate = False
        self.frameIsFrozen = False
        self.prev_frame = None
        self.last_call = 0
        self.delay = 10  # delay in seconds
        self.lock = threading.Lock()
        self.search_state = 'right'  # Drone rotates to the right initially
        self.rotation_angle = 0  # Initialize rotation angle
        self.search_time = 0  # Initialize search time
        self.rotate_search_speed = 30
        self.vertical_state = 'middle'
        self.has_tag = False
        self.big_tag = None
        self.H = None
        self.Hr = None
        self.yaw_error = 0
        self.fb_error = 0
        self.left_right_offset = 0
        self.up_down_offset = 0
        self.tracking_state = 'none'
        self.lower_threshold = 0
        self.upper_threshold = 0
        self.for_back_good = False
        self.up_down_good = False
        self.yaw_good = False
        self.left_right_good = False
        self.frame_center = None
        self.current_frame = None
        self.last_seen_time = None  # Initialize the last seen time as None
        self.last_left_right_not_good_time = None
        self.thru_frame_count = 0
        self.isLanding = False
        self.best_contour_area = 0
        self.best_contour_center = None
        self.best_contour = None
        self.land_left_right_good_for_good = False
        self.is_H_seen = False

        self.use_ids = False
        # self.tag_numbers = [2, 1, 4, 3]
        self.tag_numbers = [1, 4, 3]
        self.max_frame_count = len(self.tag_numbers)

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode(self.hud_size)

        # create update timer
        pygame.time.set_timer(USEREVENT + 1, 50)

    # main
    def run(self):
        """
        Main loop.
        Contains reading the incoming frames, the call for tracking and basic keyboard stuff.
        """

        if not self.tello.connect():
            print("Tello not connected")

        if not self.tello.set_speed(self.internalSpeed):
            print("Not set speed to lowest possible")

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")

        if not self.tello.streamon():
            print("Could not start video stream")

        frame_read = self.tello.get_frame_read()

        self.bat = self.tello.get_battery()

        self.frame_center = (self.hud_size[0] // 2, int(self.hud_size[1] // 2 + self.hud_size[1] / 4))

        # self.tracking_state = 'landing'
        # self.isLanding = True

        self.should_stop = False
        while not self.should_stop:
            # Get current frame
            self.current_frame = frame_read.frame
            current_frame = self.current_frame

            # Check if the frame has frozen
            if self.prev_frame is not None and np.array_equal(self.prev_frame, current_frame):
                # Ignore frame and set velocities to 0
                self.Move_fb_lr_up_v(0, 0, 0, 0)
                self.frameIsFrozen = True
            else:
                self.frameIsFrozen = False

            # If the frame hasn't frozen, continue with the usual processing
            self.prev_frame = current_frame
            img = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            hud_img = img.copy()
            # get output from tracking
            if self.isTracking and not self.frameIsFrozen:
                img = self.apriltags(img)
                img = self.track()

            hud_img = adjust_frame_for_april_tag(hud_img)

            # produce hud
            self.frame = self.write_hud(hud_img)
            self.frame = np.fliplr(self.frame)
            self.frame = np.rot90(self.frame)

            self.frame = pygame.surfarray.make_surface(self.frame)

            self.screen.fill([0, 0, 0])
            self.screen.blit(self.frame, (0, 0))
            pygame.display.update()

            # handle input from dronet or user
            for event in pygame.event.get():
                if event.type == USEREVENT + 1:
                    self.send_input()
                elif event.type == QUIT:
                    self.should_stop = True
                elif event.type == KEYDOWN:
                    if (event.key == K_ESCAPE) or (event.key == K_BACKSPACE):
                        self.should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == KEYUP:
                    self.keyup(event.key)

                # shutdown stream
                if frame_read.stopped:
                    frame_read.stop()
                    break

            # wait a little
            time.sleep(1 / self.FPS)

        # always call before finishing to deallocate resources
        self.tello.end()

    def track(self):

        frame = self.current_frame.copy()

        # Calculate the center of the AprilTag
        if self.bcorners is not None:
            tag_center = np.mean(self.bcorners, axis=0, dtype=np.int32)
        else:
            tag_center = None

        # Initialize PD controller parameters
        kplr = 0.065  # Proportional gain for left/right 0.1
        kdlr = 0.05  # Derivative gain for left/right 0.1

        kpud = 0.3  # Proportional gain for up/down 0.2
        kdud = 0.05  # Derivative gain for up/down 0.1

        kpfb = 0.2  # Proportional gain for for/back
        kdfb = 0.33  # Derivative gain for for/back

        kpyaw = 0.028
        kdyaw = 0.02

        # Check if the AprilTag is visible
        if tag_center is not None and self.H is not None:
            self.has_tag = True
            self.last_seen_time = time.time()  # Update the last seen time
            self.for_back_velocity = 1
            self.up_down_velocity = 0
            self.up_down_offset = 0
            self.left_right_offset = 0
            self.pitch_angle_deg = 0
            self.yaw_velocity = 0

            # Normalize the homography matrix
            H = self.H / self.H[2, 2]

            if H is not None and self.Hr is not None:
                # Calculate the percentage difference between H_new and the previous H
                percentage_diff = np.abs((H - self.Hr) / self.Hr * 100)

                # Apply the condition to ignore large changes in H
                H_updated = np.where(percentage_diff >= 10, H, self.Hr)
            else:
                H_updated = H

            self.Hr = np.round(H_updated, decimals=1)
            h1 = self.Hr[:, 0]
            h2 = self.Hr[:, 1]

            # forward backward tracking
            desired_area = 13000
            tolerance = 0.3  # 10% tolerance

            left_right_somewhat_good = False

            self.lower_threshold = desired_area - (desired_area * tolerance)
            self.upper_threshold = desired_area + (desired_area * tolerance)

            # Calculate left/right offset based on the center of the AprilTag
            self.left_right_offset = (self.frame_center[0] - tag_center[0])  # + 69
            self.left_right_offset = round(self.left_right_offset, 2)

            # Adjust up/down velocity to keep the drone centered with the AprilTag
            self.up_down_offset = self.frame_center[1] - tag_center[1]
            self.up_down_offset = round(self.up_down_offset, 2)

            # Adjust for/back velocity to keep the drone centered with the AprilTag
            self.fb_error = (desired_area - self.aparea) / 99
            self.fb_error = round(self.fb_error, 2)

            if self.lower_threshold < self.aparea < self.upper_threshold:
                self.for_back_good = True
                max_speedfb = 0
            else:
                self.for_back_good = False
                max_speedfb = 27

            if abs(self.left_right_offset) < 18:
                self.left_right_good = True
                max_speedlr = 0
            else:
                self.left_right_good = False
                max_speedlr = 17

            if abs(self.left_right_offset) < 25:
                left_right_somewhat_good = True
            else:
                left_right_somewhat_good = False

            if abs(self.up_down_offset) < 22:
                self.up_down_good = True
                max_speedud = 0
            else:
                self.up_down_good = False
                max_speedud = 17

            if abs(self.yaw_error) < 50:
                self.yaw_good = True
                max_speedyaw = 0
            else:
                self.yaw_good = False
                max_speedyaw = 14

            self.left_right_velocity = -int((kplr * self.left_right_offset - kdlr * self.left_right_velocity) * 1.1)
            self.left_right_velocity = min(max_speedlr, max(-max_speedlr, self.left_right_velocity))  # Limit speed
            if 0 < self.left_right_velocity < 7:
                self.left_right_velocity = 7
            elif -7 < self.left_right_velocity < 0:
                self.left_right_velocity = -7

            self.up_down_velocity = int(kpud * self.up_down_offset - kdud * self.up_down_velocity)
            self.up_down_velocity = min(max_speedud, max(-max_speedud, self.up_down_velocity))  # Limit speed

            # Check if they are orthogonal
            dot_product = np.dot(h1, h2) - 360

            # Use dot_product and length_difference to adjust yaw and up_down velocity
            self.yaw_error = dot_product

            self.yaw_error = np.round(self.yaw_error, 2)

            if left_right_somewhat_good:
                self.yaw_velocity = int((-kpyaw * self.yaw_error - kdyaw * self.yaw_velocity) * .9)
                self.yaw_velocity = min(max_speedyaw, max(-max_speedyaw, self.yaw_velocity))  # Limit speed
            else:
                self.yaw_velocity = 0

            if self.aparea < self.lower_threshold:
                self.for_back_velocity = int(kpfb * self.fb_error - kdfb * self.for_back_velocity * 1)
                self.for_back_velocity = min(max_speedfb, max(-max_speedfb, self.for_back_velocity))  # Limit speed
            elif self.aparea > self.upper_threshold:
                self.for_back_velocity = int(kpfb * self.fb_error + kdfb * self.for_back_velocity * 1)
                self.for_back_velocity = min(max_speedfb, max(-max_speedfb, self.for_back_velocity))  # Limit speed
            else:
                self.for_back_velocity = 0  # Maintain current position

            if self.for_back_good and self.left_right_good and self.up_down_good and self.yaw_good:
                self.Move_fb_lr_up_v(0, 0, 0, 0)
                threading.Thread(target=self.go_thru_frame).start()

        else:

            self.Move_fb_lr_up_v(0, 0, 0, 0)

            if self.last_seen_time is None or time.time() - self.last_seen_time >= 4:
                # Rotate the drone to search for the AprilTag
                if self.search_state == 'right':
                    self.rotation_angle += self.rotate_search_speed / self.FPS  # This assumes Speed is in degrees per second
                    if self.rotation_angle >= 30:  # Reached the right edge of the search range
                        self.search_state = 'left'
                else:  # self.search_state == 'left'
                    self.rotation_angle -= self.rotate_search_speed / self.FPS
                    if self.rotation_angle <= -30:  # Reached the left edge of the search range
                        self.search_state = 'right'
                        if self.vertical_state == 'down':
                            self.vertical_state = 'up'
                        elif self.vertical_state == 'middle':
                            self.vertical_state = 'down'
                        else:  # self.vertical_state == 'up'
                            self.vertical_state = 'middle'

                # Set the drone's yaw velocity based on the search state
                self.yaw_velocity = self.rotate_search_speed if self.search_state == 'right' else -self.rotate_search_speed
                self.for_back_velocity = 0
                self.left_right_velocity = 0
                self.up_down_velocity = 0
                self.up_down_offset = 0
                self.left_right_offset = 0
                self.pitch_angle_deg = 0

                # Add up-down movement
                if self.vertical_state == 'down':
                    self.up_down_velocity = -10
                elif self.vertical_state == 'up':
                    self.up_down_velocity = 25
                else:  # self.vertical_state == 'middle'
                    self.up_down_velocity = 0

                self.left_right_velocity = 0

        return frame

    def landing(self):
        # Constants for PD controller
        KPlr = 0.075  # Proportional gain
        KDlr = 0.04  # Derivative gain

        # Capture frame from the drone's camera
        yoloframe = self.tello.get_frame_read().frame.copy()

        yoloframe = cv2.cvtColor(yoloframe, cv2.COLOR_BGR2RGB)

        # Preprocess frame for inference
        yoloimg = yoloframe.copy()

        # Inference
        results = self.Yolomodel(yoloimg)

        # print(f"left right good for good: {self.land_left_right_good_for_good}")
        # print(f"last seen time: {self.last_seen_time}")
        # Process and display the results

        self.is_H_seen = False

        for detection in results.xyxy[0]:
            self.detection = detection
            x, y, w, h, confidence, class_id = detection.tolist()
            class_name = self.Yolomodel.names[int(class_id)]

            # Check if the detected object is 'H'
            if class_name == 'H' and confidence > .3:
                self.is_H_seen = True  # Update this value when 'H' is detected

            # Draw bounding box and label on the frame
            cv2.rectangle(yoloframe, (int(x), int(y)), (int(w), int(h)), (255, 0, 0), 2)
            cv2.putText(yoloframe, f'{class_name} {confidence:.2f}', (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0), 2)

            # Calculate object frame center
            object_center_x = int((x + w) // 2)
            object_center_y = int((y + h) // 2)

            break

        if self.land_left_right_good_for_good:
            # left right movement is good for good
            # (ive been working for 10 hours i cant word it better than this)

            self.left_right_velocity = 0
            self.for_back_velocity = 15

            if self.is_H_seen:
                self.last_seen_time = time.time()  # Update the last seen time
            else:
                self.for_back_velocity = 0
                if self.last_seen_time is None or time.time() - self.last_seen_time >= 4:
                    self.tello.move_forward(40)

                    time.sleep(3.5)

                    self.for_back_velocity = 0
                    self.tello.land()
                    self.tracking_state = 'none'
                    self.tello.streamoff()
                    self.tello.end()

                    return yoloframe

            return yoloframe

        if self.is_H_seen:

            self.Move_fb_lr_up_v(0, 0, 0, 0)

            if self.last_left_right_not_good_time is not None and time.time() - self.last_left_right_not_good_time >= 5:
                self.land_left_right_good_for_good = True

            self.last_seen_time = time.time()  # Update the last seen time

            # Calculate object area
            object_area = np.round(w * h, decimals=1)

            # puts info on screen
            cv2.putText(yoloframe, str(object_area), (object_center_x, object_center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 3)

            cv2.arrowedLine(yoloframe, (self.midx, self.midy),
                            (object_center_x, object_center_y),
                            (0, 255, 0), 4)

            # Calculate errors for PD controller
            self.left_right_offset = object_center_x - self.frame_center[0]
            self.left_right_offset = np.round(self.left_right_offset, 2)

            self.left_right_good = False

            if abs(self.left_right_offset) < 30:
                self.left_right_good = True
                max_speedlr = 0
            else:
                self.last_left_right_not_good_time = time.time()

                self.left_right_good = False
                max_speedlr = 20
            if self.left_right_good:
                print("good")
            else:
                self.left_right_velocity = int((KPlr * self.left_right_offset - KDlr * self.left_right_velocity) * 1.05)
                self.left_right_velocity = min(max_speedlr,
                                               max(-max_speedlr, self.left_right_velocity))  # Limit speed
                if 0 < self.left_right_velocity < 6:
                    self.left_right_velocity = 6
                elif -6 < self.left_right_velocity < 0:
                    self.left_right_velocity = -6
        else:
            # no H in sight

            self.Move_fb_lr_up_v(0, 0, 0, 0)

            if self.last_seen_time is None:
                self.last_seen_time = time.time()

                return yoloframe

            if self.last_seen_time is not None and time.time() - self.last_seen_time >= 3:

                if self.search_state is None:  # This will be the case the first time this runs
                    self.search_state = 'right'  # Or 'left', depending on where you want to start the search
                    self.rotation_angle = 0
                if self.search_state == 'right':
                    self.rotation_angle += self.rotate_search_speed / self.FPS  # This assumes Speed is in degrees per second
                    if self.rotation_angle >= 12:  # Reached the right edge of the search range
                        self.search_state = 'left'
                else:  # self.search_state == 'left'
                    self.rotation_angle -= self.rotate_search_speed / self.FPS
                    if self.rotation_angle <= -12:  # Reached the left edge of the search range
                        self.search_state = 'right'

                # Set the drone's yaw velocity based on the search state

                self.yaw_velocity = self.rotate_search_speed if self.search_state == 'right' else -self.rotate_search_speed

        return yoloframe

    def average_color(self, frame, contour):
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_val = cv2.mean(frame, mask=mask)

        return mean_val[:3]  # Exclude alpha channel

    def get_color_range(self, center_color, tol=0.1):
        # Converts the center color and tolerance from a [0, 255] scale to a [0, 1] scale.
        center_color = np.array(center_color) / 255.0
        tol = tol / 255.0

        # Calculate the lower and upper bounds, making sure they remain in the valid range [0, 1].
        lower_color = np.maximum(0, center_color - tol)
        upper_color = np.minimum(1, center_color + tol)

        # Converts the color ranges back to a [0, 255] scale for use with OpenCV.
        lower_color = (lower_color * 255).astype(np.int32)
        upper_color = (upper_color * 255).astype(np.int32)

        return lower_color, upper_color

    def go_thru_frame(self):
        with self.lock:
            current_time = time.time()
            if current_time - self.last_call < self.delay or self.frameIsFrozen:
                return
            self.last_call = current_time

        self.isTracking = False
        self.tracking_state = 'thru frame'
        self.isGoingThruGate = True
        self.for_back_velocity = 0
        self.yaw_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.Move_fb_lr_up_v(0, 0, 0, 0)
        print("started going thru gate")

        # await asyncio.sleep(1)
        time.sleep(1)
        # self.Move_fb_lr_up_v(0, 0, 0, 0)
        # self.tello.move_up(20)
        # print("going up")
        # self.tello.send_command_without_return("up 35")

        # await asyncio.sleep(3)
        # time.sleep(3)
        # self.tello.move_forward(60)

        print("going forward")
        self.tello.send_command_without_return("forward 120")
        # await asyncio.sleep(3.0)
        time.sleep(5)

        self.isTracking = True
        self.tracking_state = 'tracking'
        self.for_back_velocity = 0
        self.yaw_velocity = 0
        self.up_down_velocity = 0
        self.left_right_velocity = 0
        self.Move_fb_lr_up_v(0, 0, 0, 0)
        self.isGoingThruGate = False
        self.vertical_state = 'middle'
        self.plus_one_frame_count()

        print("tracking now!")

    # input Method
    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_w:  # set forward velocity
            self.isTracking = False
            self.for_back_velocity = self.controll_params['Speed']
        elif key == pygame.K_s:  # set backward velocity
            self.isTracking = False
            self.for_back_velocity = -self.controll_params['Speed']
        elif key == pygame.K_a:  # set left velocity
            self.isTracking = False
            self.left_right_velocity = -self.controll_params['Speed']
        elif key == pygame.K_d:  # set right velocity
            self.isTracking = False
            self.left_right_velocity = self.controll_params['Speed']
        elif key == pygame.K_r:  # set up velocity
            self.isTracking = False
            self.up_down_velocity = self.controll_params['Speed']
        elif key == pygame.K_f:  # set down velocity
            self.isTracking = False
            self.up_down_velocity = -self.controll_params['Speed']
        elif key == pygame.K_e:  # set yaw clockwise velocity
            self.isTracking = False
            self.yaw_velocity = self.controll_params['Speed']
        elif key == pygame.K_q:  # set yaw counter clockwise velocity
            self.isTracking = False
            self.yaw_velocity = -self.controll_params['Speed']
        elif key == pygame.K_b:  # takeoff
            self.tello.send_command_without_return("takeoff")
            self.send_rc_control = True
        elif key == pygame.K_LSHIFT:  # land
            self.isTracking = False
            self.tracking_state = 'none'
            self.tello.land()
            self.send_rc_control = False
        elif key == pygame.K_p:
            self.plus_one_frame_count()
        elif key == pygame.K_BACKSPACE:  # emergency shutdown
            self.tello.send_command_without_return("land")
            self.isTracking = False
            self.send_rc_control = False
            self.should_stop = True
        elif key == pygame.K_t:  # arm tracking
            self.isTracking = True
            self.tracking_state = 'tracking'
            self.for_back_velocity = 0
            self.yaw_velocity = 0
            self.up_down_velocity = 0
        elif key == pygame.K_c:  # get new color
            self.set_color(self.central_color)
            self.for_back_velocity = 0
            self.yaw_velocity = 0
            self.up_down_velocity = 0
        elif key == pygame.K_HASH:  # switch parameters
            if self.current_parameter == 0:
                self.current_parameter = 1
            else:
                self.current_parameter = 0
        elif key == pygame.K_PLUS:  # raise current parameter
            what = self.param_keys[self.current_parameter]
            if self.controll_params[what] < self.controll_params_m[what] - 0.01:
                self.controll_params[what] = self.controll_params[what] + self.controll_params_d[what]
                if (what == 'Color'):
                    self.reset_color()
        elif key == pygame.K_MINUS:  # lower current parameter
            what = self.param_keys[self.current_parameter]
            if self.controll_params[what] > 0.01:
                self.controll_params[what] = self.controll_params[what] - self.controll_params_d[what]
                if (what == 'Color'):
                    self.reset_color()

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """

        if key == pygame.K_w or key == pygame.K_s:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_r or key == pygame.K_f:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_q or key == pygame.K_e:  # set zero yaw velocity
            self.yaw_velocity = 0

    def send_input(self):
        """ Update routine. Send velocities to Tello."""
        # print("V: " + str(self.for_back_velocity) + "; Y: " + str(self.yaw_velocity))
        if self.send_rc_control and (self.isTracking or self.isLanding or self.tracking_state == 'landing'):
            try:
                self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                           self.yaw_velocity)
            except Exception as e:
                print(f"An error occurred while sending control commands to the drone: {e}")

    # Helper Methods

    def update_color(self, val):
        """
        Adjusts the currently tracked color to input.
        """
        if (cv2.mean(val) != 0):
            for i in range(0, 2):
                self.current_color[i] = clamp(val[i],
                                              self.color_lower[self.color_keys[self.controll_params['Color']]][i],
                                              self.color_upper[self.color_keys[self.controll_params['Color']]][i])

    def set_color(self, val):
        self.current_color = np.array(val)
        print(val)

    def reset_color(self):
        self.current_color = np.array(self.color_lower[self.color_keys[self.controll_params['Color']]]) + np.array(
            self.color_upper[self.color_keys[self.controll_params['Color']]])
        for i in range(0, 3): self.current_color[i] = self.current_color[i] / 2

    def write_hud(self, frame):
        """Draw drone info and record on frame"""
        stats = ["TelloTracker"]
        stats.append("battery: " + str(self.bat) + "%")
        if self.isTracking:
            stats.append("Tracking active.")
            stats.append("Speed: {:03d}".format(self.controll_params['Speed']))
            # stats.append("Color: " + self.color_keys[self.controll_params['Color']])
            frame = self.apriltags(frame)
            frame = self.draw_arrows(frame)
        else:
            stats.append("Tracking disabled.")
            img = cv2.circle(frame, (self.midx, self.midy), 10, (0, 0, 255), 1)

        if self.isLanding:
            frame = self.landing()

        if self.frameIsFrozen:
            stats.append("frame frozen")

        stats.append(f"best contour area: {self.best_contour_area}")
        if self.has_tag and self.H is not None:
            self.has_tag = self.has_tag
        stats.append(self.param_keys[self.current_parameter] + ": {:4.1f}".format(
            self.controll_params[self.param_keys[self.current_parameter]]))

        for idx, stat in enumerate(stats):
            text = stat.lstrip()
            cv2.putText(frame, text, (0, 30 + (idx * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0), lineType=30)

        # Draw speeds on the top right side of the frame
        speed_text1 = f"For/Back Speed: {self.for_back_velocity}"
        speed_text2 = f"Yaw Speed: {self.yaw_velocity}"
        speed_text3 = f"Up/Down Speed: {self.up_down_velocity}"
        speed_text4 = f"Area: {self.aparea} {self.for_back_good}"
        speed_text5 = f"left/right offset: {self.left_right_offset}"
        speed_text6 = f"yaw erroe: {self.yaw_error} {self.yaw_good}"
        speed_text7 = f"left/right speed: {self.left_right_velocity} {self.left_right_good}"
        speed_text8 = f"tracking state: {self.tracking_state}"
        speed_text9 = f"left/right offset: {self.left_right_offset}"
        speed_text10 = f"is H seen: {self.is_H_seen}"
        speed_text11 = f"good for good: {self.land_left_right_good_for_good}"
        speed_text12 = f"pitch: {self.pitch_angle_deg}"
        speed_text13 = f"ud : {self.up_down_offset}"
        speed_text14 = f"lr : {self.left_right_offset}"

        cv2.putText(frame, speed_text1, (self.hud_size[0] - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, speed_text2, (self.hud_size[0] - 250, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, speed_text3, (self.hud_size[0] - 250, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, speed_text4, (self.hud_size[0] - 250, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, speed_text6, (self.hud_size[0] - 250, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, speed_text7, (self.hud_size[0] - 250, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, speed_text8, (self.hud_size[0] - 250, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, speed_text9, (self.hud_size[0] - 250, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, speed_text10, (self.hud_size[0] - 250, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, speed_text12, (self.hud_size[0] - 250, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, speed_text13, (self.hud_size[0] - 250, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, speed_text14, (self.hud_size[0] - 250, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, speed_text5, (self.hud_size[0] - 550, 540),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, speed_text11, (self.hud_size[0] - 550, 570),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        return frame

    def draw_arrows(self, frame):
        """Show the direction vector output in the cv2 window"""
        # cv2.putText(frame,"Color:", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)
        cv2.arrowedLine(frame, (self.midx, self.midy),
                        (self.midx + self.xoffset, self.midy - self.yoffset),
                        (255, 0, 0), 5)

        return frame

    def apriltags(self, frame):
        frame = frame.copy()

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags
        tags = self.tag_detector.detect(gray_frame, estimate_tag_pose=True, camera_params=tuple(self.t_camera_params),
                                        tag_size=self.tagsize)

        if self.use_ids:
            # Filter tags
            tags = [tag for tag in tags if tag.tag_id == self.tag_numbers[self.thru_frame_count]]

        # Draw red frames around detected tags and display tag IDs
        for tag in tags:
            corners = np.array(tag.corners, dtype=np.int32)
            # Convert frame to numpy.ndarray
            frame = np.array(frame)
            cv2.polylines(frame, [corners], True, (0, 0, 255), 2)

            center = np.mean(corners, axis=0, dtype=np.int32)
            cv2.putText(frame, str(tag.tag_id), (center[0] - 40, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

            # Retrieve the pose of the tag
            pose_R = tag.pose_R
            pose_t = tag.pose_t
            pose_err = tag.pose_err

            # Draw the axis lines on the frame
            axis_length = 50  # Adjust as needed
            origin = tuple(center)
            self.aporigin = origin

            self.x_axis_end = (int(origin[0] + axis_length * pose_R[0, 0]), int(origin[1] + axis_length * pose_R[1, 0]))
            self.y_axis_end = (int(origin[0] + axis_length * pose_R[0, 1]), int(origin[1] + axis_length * pose_R[1, 1]))
            self.z_axis_end = (int(origin[0] + axis_length * pose_R[0, 2]), int(origin[1] + axis_length * pose_R[1, 2]))

            cv2.line(frame, self.x_axis_end, origin, (255, 0, 0), 2)  # Draw x-axis (blue)
            cv2.line(frame, self.y_axis_end, origin, (0, 255, 0), 2)  # Draw y-axis (green)
            cv2.line(frame, self.z_axis_end, origin, (255, 165, 0), 2)  # Draw x-axis (orange)

        # Find the biggest tag and draw a green border around it
        if tags:
            biggest_tag = max(tags, key=lambda x: x.tag_id)
            self.big_tag = biggest_tag
            self.H = biggest_tag.homography
            self.bcorners = np.array(biggest_tag.corners, dtype=np.int32)
            self.bcorners = np.ascontiguousarray(self.bcorners)  # Ensure contiguous array
            # Draw orange line connecting center of screen and center of AprilTag
            cv2.line(frame, tuple(center), self.frame_center, (255, 165, 0), 2)

            # Draw yellow dot at the center of the screen
            cv2.circle(frame, self.frame_center, 3, (0, 255, 255), -1)

            # Draw purple dot at the center of the AprilTag
            cv2.circle(frame, tuple(center), 3, (255, 0, 255), -1)
            # Calculate the area of the AprilTag
            self.aparea = cv2.contourArea(self.bcorners)
            # Print the ID and area of the AprilTag
            text = f"A2:{self.aparea:.2f}"

            cv2.polylines(frame, [self.bcorners], True, (0, 255, 0), 2)
            cv2.putText(frame, text, (center[0] - 40, center[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            self.bcorners = None
            self.big_tag = None
            self.H = None
        return frame

    def clamp(self, n, smallest, largest):
        return max(smallest, min(n, largest))

    def Move_fb_lr_up_v(self, for_back, left_right, up_down, yaw):
        self.for_back_velocity = for_back
        self.left_right_velocity = left_right
        self.up_down_velocity = up_down
        self.yaw_velocity = yaw

    def plus_one_frame_count(self):
        self.thru_frame_count += 1

        if self.thru_frame_count == self.max_frame_count:
            self.isTracking = False

            print("going down")
            # self.tello.send_command_without_return("down 30")
            self.tello.send_command_without_return('forward 25')
            time.sleep(4)
            self.tello.send_command_without_return("down 50")

            time.sleep(5)

            self.tello.send_command_without_return("back 20")
            time.sleep(3)

            print("start landing")
            self.isLanding = True
            self.tracking_state = 'landing'

            return


def adjust_frame_for_april_tag(frame):
    frame = adjust_gamma(frame, 1.4)
    frame = apply_brightness_contrast(frame, 20, 45)
    return frame


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
