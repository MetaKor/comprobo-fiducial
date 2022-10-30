
# os allows us to find the file directory of where our training image is stored
import os

# import opencv to give us access to a wide range of image processing tools
import cv2

# import numpy for a wide range of mathematical tools
import numpy as np


class object_detection():

    # init initializes any code contained within it when an instance is created
    # all global variables & constants are defined here, stored in self
    def __init__(self):
        # define where the training image will be stored
        self.image_dir = os.getcwd() + "/Images/train.jpg"
        # define a variable to store the threshold at which we keep corners
        self.corner_threshold = 0.0
        # define a variable to check how distinct a corner is (no close second match)
        self.close_match_threshold = 1.0


    def find_corners(self, image):
        # takes an image and returns the corners within it

        # convert the image to black and white as we only care for gradient shift
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find the corners in the images (also commonly called key points)
        corners = cv2.SIFT_create().detect(image_bw)

        # now filter only corners that are above the predefined threshold
        # this is the response and it indicates how strong SIFT_create considers
        # a corner to be based on how much gradient shift it has in two axes
        corners = [corner for corner in corners if corner.response > self.corner_threshold]

        return corners

    def corner_bounds(self, corners):
        # find the four edges of the corners
        x = self.find_x(corners) # find the x coordinates of each corner
        y = self.find_y(corners) # find the y coordinates of each corner

        # find the min and max of each
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)

        # create a list to store the x & y coordinates
        x = [x_min, x_min, x_max, x_max]
        y = [y_min, y_max, y_min, y_max]

    def find_x(self,corners):
        # takes a list of corners and returns a list of their x pixel coordinates
        # pt is a tuple that stores the x,y position of each corner
        x = [corner.pt[0] for corner in corners]

        return x

    def find_y(self, corners):
        # take a list of corners and returns a list of their y pixel coordinates
        # pt is a tuple that stores the x,y position of each corner
        y = [corner.pt[1] for corner in corners]

        return y

if __name__ == '__main__':
    od = object_detection()

    # define where to receive the video stream
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        # load the live video feed
        ret, frame = cap.read()
        if frame is None:
            continue

        # find corners in the frame and draw a circle wherever they are
        corners = od.find_corners(frame)

        for corner in corners:
            # draw a circle on the video frame where each corner is
            cv2.circle(frame,(int(corner.pt[0]),int(corner.pt[1])), 10, (0,0,255), -1)
        frame = np.array(cv2.resize(frame,
                                    (frame.shape[1]//1,
                                     frame.shape[0]//1)))
        cv2.imshow('Video Feed', frame)
        # if the user presses esc, terminate the process
        cv2.waitKey(27)
    cv2.destroyAllWindows()
