# Detecting-Circles-In-Image
 
The Hough transform is a feature extraction technique used in computer vision and digital image processing. The purpose of the technique is to find imperfect instances of objects within a certain class of shapes by a voting procedure. In this assignment, I started by using any edge detection method to obtain edge points from an image, and then I implemented Hough Transform to detect circles.

All implementation details and examples in the report file. 

You can run the code prom main.py. There is no need to give argument. 

You can change radius_min, radius_max and threshold values. 

There are 5 functions: 

main() : It do hough transorm for all images in dataset/Images folder and calculate Intersection over Union (IoU) with dataset/GT txt files. 

CannyEdgeDetection() : It does Gaussian Blur and Canny Edge Detection to the image.

CreateAccumulatorList() : Is creates accumulator list. 

DetectHoughCircles() : It draws founded circles to input image.

areaOfIntersection() : It calculates union area of two circles.

main() -> CannyEdgeDetection() -> CreateAccumulatorList() -> DetectHoughCircles()
