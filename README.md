## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup/car_notcar.jpg
[image2]: ./writeup/HOG.jpg
[image3]: ./writeup/search_windows.jpg
[image4]: ./writeup/search_windows2.jpg
[image5]: ./writeup/heatmap.jpg
[image6]: ./writeup/labels.jpg
[image7]: ./writeup/bounding_boxes.jpg
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Reading and Splitting dataset.

I started by reading in all the `cars` and `notcars` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

then I tried to manually split the `cars` and `notcars` images into train and test sets, taking into consideration that successive images in the dataset are very similar, so I made sure that after taking a batch of train images, I made a jump of 20 images or so before taking a batch of test images, and so on...(Cell #4), but it didn't make a noticeble difference from randomly splitting the test and train images, so I settled for using sklearn radom `train_test_split` (Cell #14).

#### 2. Explain how (and identify where in your code) you extracted HOG features from the training images.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 3. Explain how you settled on your final choice of HOG parameters.

I tried the `RGB`, `HSV`, `HLS`, `YUV`, `LUV` & `YCrCb`, for extracting the HOG features, but the 'YCrCb' lead to the best accuracy and results, then I tried different values for `orientations`, 6, 9, & 12, it seems that 12 is too many and might lead to overfitting, and 6 won't detect all the features of the car leading to underfitting, so I settled for 9 orientations which gave the best accuracy.

#### 4. HOG & Color Features Extraction.

I then extracted the HOG features, (Cell #7), by using the `extract_hog_features` function in `helpers.py`, which in turn calls the `get_hog` function (also in `helpers.py`) for each channel after reading multiple files and converting the colorspace, it is worth noting here that setting the block normalization method to `L1` instead of `L2-Hys` for the sklearn hog function, lead to dramatically improved results and much less false positives.

Then I exctracted the spatial and histogram color features, using the `YCrCb` colorspace as well, (Cell #10), using the `extract_color_features` function in `helpers.py`, which uses the `bin_spatial` and `color_hist` functions (also in `helpers.py`), after reading multiple files and converting the colorspaces, then it concatenates the features from both functions, it is worth noting here that 16 x 16 pixels for the spatial color features, gave better (and faster) results than 32 x 32, which shows that 32 x 32 might be too specific for each car and leads to overfitting, the same applies to the histogram bins, where 16 bins gave better results than 32 bins.

Then I combined both HOG and Color features (Cell #12).

#### 5. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In (Cell #17) I trained the model using a linearSVC classifier after splitting and scaling the data (Cell #14) (using Sklearn `train_test_split` and `StandardScaler` functions), I experimented with different values for the C parameter, using the `GridSearchCV` method, but it didn't make much difference, so I opted for the default value (C=1.0) which lead for an accuracy above 98%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use the HOG subsampling method to increase efficiency and reduce the prediction time, the `find_cars` function defined in (Cell #19) extracts the HOG features for the whole region of interest in one shot, (but the color features for each window), and it uses the `scale` parameter and the `cells_per_step` to define the final size and positions of the of the search windows within the region of interest (which is defined by the x & y start & stop parameters), then it extracts the features for each window and does the prediction on it, using the pretrained model, I tested the `decision_function` with different thresholds, but it didn't make much difference, so I sticked with the `predict` function.

The step size was defined as 2 cells per step, which gave the best results, and for the scale, 3 different regions were defined (Cell #10) large windows in the bottom (2.0 scale) and medium sized windows in the middle (1.5 scale) and small sized windows on the top (1.0 scale) for the smaller cars further away, the `find_cars` function is called multiple times for each region and the bounding boxes for each call is drawn on the output image and added to the bounding boxes list.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color (in YCrCb as well) Here is an example of the positive detections:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap (defined in Cell #21) and with the help of python dequeue functions, I combined all the heatmaps for the last 10 frames of the video at any point in time (except for the first  10 frames), and applied a threshold (of 8) to the compiled heatmap to differentiate the spurious false positives that will appear in a one frame and disapper in the next from the more persistent true vehicle detections. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each detected blob.  

Here's an example of the heatmap of a video frame:

![alt text][image5]

Here is the result of the `scipy.ndimage.measurements.label()` function on the same video frame:

![alt text][image6]

And here is the result showing the final bounding boxes of this frame:

![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* The Pipeline misses the white car some of the time, and there are some false positives, especially in the shadowy areas, I think the provided dataset needs to be augmented, by adding noise and changing the brightness of the images, or adding more samples of cars (especially white).

* The detected vehicles need to be tracked in order to predict where they will be in the next frames, which will make the bounding box size consistent, and it's movement smooth, and will keep track of the vehicles even if the detection is missed in a couple of frames, also if all vehicles in sight are tracked, it will prevent the algorithm from combining 2 cars in 1 bounding box when they are near each other.
