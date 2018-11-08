# **Project: Finding Lane Lines on the Road-Advanced** 
### This project is advanced version of the simple [Lane Line Detection](https://github.com/sanket1012/FindingLaneLines) algorithm uses computer vision and image processing tecniques to detect the Lane Lines that are in the visible range of the car. Following explained is an overview of different steps involved in the building block of the entire Image Pipeline, potential shortcomings in the current method and area of improvement.

## Overview
**The goals / steps of this project are the following:**

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/undistorted/cal1.jpg
[image2]: ./test_outputs/undistorted_test6.jpg
[image3]: ./test_outputs/thresholded_image_test6.jpg
[image4]: ./test_outputs/warped_image_test3.jpg
[image5]: ./test_outputs/polyfitted.png
[image6]: ./test_outputs/final_image_test3.jpg

## Brief Explaination of each step:

### Camera Calibration and Distortion Correction

- There are chances that video feed from camera have different distortion effects in the image. Distortions like Radial Distortion and Tangential Distortion. So in our first step we will calibrate our camera so that any such type of image will be taken care of in future.
- Step 1 of the [code](./Advanced Lane Lines.ipynb) performes this step. To calibrate camera we will convert our 3D points in World Space to 2D point in Image Space. These points are assigned to ObjPoints (3D points) and ImgPoints (2D points). 
- Test images of Chess Board are used to perform camera calibration. Here we will use the corners found in the Chess Board as the ImgPoints and XY coordinates(z=0 as assuming chessboards are are on flat surface) to ObjPoints. [cv2.findChessboardCorners()](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)
- After having these points we will perform camera calibration using [cv2.calibrateCamera()](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html). This will provide us with Calibration Matrix('mtx'), Distortion Coefficient('dist'), psition of Camera in real world space('rvecs','tvecs'), which will be necessary for performing distortion correction.
- In next step we will use above obtained data to perform undistortion of test images. [cv2.undistort()](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html)
- Below you can see how the distorted image of Chess Board was corrected. You can find all other undistorted images [here](./camera_cal/undistorted)
![alt text][image1]
- Now once we have our camera calibrated we will apply distortion correction to our Lane Line test images, you can find all other undistorted images for lane lines [here](./test_outputs), below is one example:
![alt_text][image2]

### Thresholding and creating Binary Image:
- we will make use of Gradient Operators and Color Spaces and based on threshold select pixels based on gradient strength and particular color space.
- Sobel Operator is used to perform gradient on the undistorted image. [cv2.Sobel()](https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html)
- Other than RGB there are different color spaces like HSV (Hue, Saturation, Value) and HLS (Hue, Lightness, Saturation). We will use HLS space and consider S Channel of the color space as it eliminates the effect of shadows in the image and retains original color under shadows. Also, one can use different color spaces to overcome any particular difficulty.
    - S-channel of HLS colorspace is good to find the yellow line and in combination with gradients, it can give you a very good result.
    - R-channel of RGB colorspace is pretty good to find required lines in some conditions.
    - L-channel of LUV colorspace for the white line.
    - B-channel of LAB colorspace may be good for the yellow line.
- Here we will also perform gaussian bluring, for smoothening of any noise or sharp edges, and Opening Morphological Operation, which is useful in closing small holes inside the foreground objects, or small black points on the object [cv2.morphologyEx()](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)
![alt text][image3]

### Perspective Transform:
- Unlike simple lane line detection instead of canny edge detection, we will take a birds eye view of the lanes in the image.
- This can be done by performing perspective transform on the image by specifying source and destination points. [cv2.getPerspectiveTransform](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html)
```python
# Specify 4 source points in the image on which Perspective transform is to be performed
src = np.float32([[(imshape[1]*0.45, imshape[0]*0.65),
                   (imshape[1]*0.6, imshape[0]*0.65),
                   (imshape[1]*0.95,imshape[0]),
                   (imshape[1]*0.15,imshape[0])]])

# Specify 4 destination points
dst = np.float32([[325,0],[1000,0],[900,720],[300,720]])

```
![alt text][image4]

###  Detect lane pixels and fit to find the lane boundary:
- To find the pixels related to Lane Lines in the Perspective Transformed image we will perform Histogram based search algorithm.
- By finding the histogram, it will provide with points where peak of histogram is high indicating as Lane Line pixel.
- Applying a sliding window across these pixels and later on fitting a curve across these pixels will provide us with a lane line curve. [np.polyfit()](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.polyfit.html)
![alt text][image5]

### Computing Radius of Curvature and Car Position:
- To get the redius in terms of real world, convert x and y from pixels space to meters:
```python
    ym_per_pix = 3.048/100 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7/378 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
```
- Radius of curvature of the lanes can be found by using:
R<sub>curve</sub> = (1 + (2.A.Y + B)<sup>2</sup>) <sup>3/2</sup> / 2.A
- And then you can get the average lane curvature of the two lanes.
- For car position, you know that camera is on the center of the car. Idealy, the position of the car is in the middle of the two lane line.
- so by finding the difference between center of the image and middle point of the two lanes we can find the position of the car from its center location.

### Final Output:
- Once all the blocks of pipeline are executed the final output will look something like this:
![alt text][image6]
- [Here](./project_video_output.mp4) is the complete operation implemented on a video.

## Problems / issues in implementation of this project:
- In this implementation we are fitting the polynomial based on Histogram based search algorithm. The performance can be improved by using different more Advanced Lane detection algorithms. Also there only few criterias used for validation of line correctness, performance can be increased by utilizing validation criteria like :
- lane width is in defined borders
- lane lines have the same concavity
- lane curvature, distance from the center, polynomial coefficients and so on.. don't differ a lot from the same values from the previous frame

- [Here](./challeneg_video_output.mp4) you can see how above issues causes the deflections in the lane line detection.
