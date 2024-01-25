This is a code that was used for FIRA 2023 competition.
The code uses apriltag and yolo5 (to train a model that detects a Blue Helli pad sign) libraries.
It's written so it makes the drone take off and search for certain number of apriltags that are attached to frames. 
The drone then searches for apriltangs and after detecting them aligns and centers itself with it so it can go through the frames with a simpel up and forward movement.
After going through all frames the drone looks for the Blue H sign and lands on it by calculating the frame sie around the size even the H sing was last visible ehwn moving towards it.
