# comprobo-fiducial
By Luke Raus & Phillip Post

## Overview
In this project, we are interested in the problem of localizing a camera in 3D space with respect to an object in its field of view. Using the pixel information captured in the camera frame to determine geometric information in the world frame is one of the most common use-cases for computer vision, with notable applications for object tracking and localization in robotics. This task is very difficult without any prior knowledge of the geometric characteristics of the world frame, such as using a camera feed of arbitrary objects of unknown shape and size. However, when dealing with an object of pre-determined (visual) geometry, i.e. a fiducial: it become the fairly well-studied Perspective-n-Point (PnP) pose computation problem, with algorithmic solutions widely exploited by visual fiducial systems like AprilTags and ArUco markers which form a quintessential component of the modern roboticist's toolkit. We are interested in learning how these systems work by attempting to build a simple fiducial system from the ground up and using it to localize our camera with respect to the fiducial by solving the PnP problem.

![Object in NEATO frame](/media/neato_frame.png)


## Challenge One: 
### How to consistently find the four exterior corners of an object irrespective of perspective?

This was one of the most challenging aspects of our project as photogrammetry is reliant on having a at least four consistent points defined from the object. At first, we attempted to use the standard close match and edge threshold within the given code to achieve this. However, we ran into the issue where making the filters too stringent would starve our algorithm of data to consistently update position, but lowering these thresholds would only lead to a flood of inconsistent reference points. 

Eventually, we came to terms that the given methods would not be enough to achieve our aims, so we explored statistical methods that could consistently provide us anchor four points on the image. The basic premise is we wanted to use an algorithm that could take a large set of data and then group it based on the datapoints proximity to their nearest neighbors. We settled on using the K-mean approach as we could dictate how many distinct groups it chose and it provided the center point of these groups to boot. Figure 1 provides an illustration of how this relatively works with each colored circle representing a group.

![K-means clustering](/media/clustering.png)
Figure 1. Example of K-mean identifying distinct groups of points

In this case, our points would be the matching corners our code identifies. Therefore, any extraneous corners that are matched outside of these groupings are thrown out. This was very convenient as dictating the number of groups ensured that no matter what we would only have four matches to feed in later in our code. The only downside encountered with the approach is that the K-mean algorithm would break if less than 5 points were provided. This makes sense as it is impossible to perform groupings if there are not enough data points. This was resolved by adding a simple if statement before the K-mean part of the code to ensure that it would only be performed if there was enough data.


## Challenge Two:
### How to identify correct quadrilateral from corners?

Even with 4 corners successfully identified, we are not quite ready to pass the data to a PnP solver. We still need to identify the order in which these 4 corners are traversed to form the boundary of the quadrilateral of interest; for example, we would not want to assert that a pair of opposite corners are actually adjacent.