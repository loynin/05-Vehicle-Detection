One up one a time, when I was watching Jamebon movie, I saw a car drive itself to save Jamebon. It was amazed me to wonder is it really there is a car that is really smart like that? Now, I am learning to build this car and I am exited about this opportunity and thrilt to use this knowledge to build my own smart car which is better than Jambon's car.

In this project, I will develop a software that a self-driving car can detect object sourounding it so it better understand the world around it. Here, let begin...

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

##### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In order to obtain the training data features for the model, I used vehicle and non-vehicle data provided by the udacity project. The following steps are the process of how to get the nessary data for the model:
1. Reading raw data and loading it into array (this is in the block code #4 in the Jupyter notebook). After reading I have visualizing data and below are the result pictures: <img src="https://github.com/loynin/05-Vehicle-Detection/blob/master/output_images/visual_8images.png" with="600">
2. In order to extract features data, I used function ```extract_features()```. extract_feature function consists of three parts are: 
* Applying color space
* Applying spatial feature
* Applying HOG feature

**Here are the example of pictures after applying the HOG features:**
<img src="https://github.com/loynin/05-Vehicle-Detection/blob/master/output_images/HOG_8_images.png" width="800">

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters such as color_space, orient and spatial_size but most of the them provide inaccurate result in detecting car. After couple days of trying different set of parameters, I have decided to use the following values as they provide the most accurated detection the vehicles on the video. Here are the used parameters:
- color_space = 'GRAY' # Can be GRAY, RGB, HSV, LUV, HLS, YUV, YCrCb
- orient = 8  # HOG orientations
- pix_per_cell = 16 # HOG pixels per cell
- cell_per_block = 1 # HOG cells per block
- hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
- spatial_size = (16, 16) # Spatial binning dimensions
- hist_bins = 16    # Number of histogram bins
- spatial_feat = False # Spatial features on or off
- hist_feat = False # Histogram features on or off
- hog_feat = True # HOG features on or off

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using SVC classifier. The processes of training the model as following:
1. Extract data features from raw car and non-car dataset (code block [9])
2. Combined data features of car and non-car data 
```X = np.vstack((car_features, notcar_features)).astype(np.float64)```
3. Defined label vector 
```y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))```
4. Split data into train and test data set 
```X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)```
5. Train by using SVM model ```svc = SVC ()```
6. By using 8 orientations 16 pixels per cell and 1 cells per block, I got **98.48%** of Test Accuracy.  

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used three windows sliding in order to detect for vehicles. These three window are:
- Window 1: x_start_stop=[None,None], y_start_stop=[400,640],xy_window=[128,128], xy_overlap=(0.5,0.5)
- Window 2: x_start_stop=[32,None], y_start_stop=[400,600],xy_window=[96,96], xy_overlap=(0.5,0.5)
- Window 3: x_start_stop=[412,1280], y_start_stop=[390,540],xy_window=[80,80], xy_overlap=(0.5,0.5)
Here are the pictures to present these windows:
<img src="https://github.com/loynin/05-Vehicle-Detection/blob/master/output_images/sliding_windows_image.png" width="800">


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using grayscale color-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

1. Heat map image illustration:
<img src="https://github.com/loynin/05-Vehicle-Detection/blob/master/output_images/heat_map_images.png" width="800">

2. Here is a result of pipeline image illustration:
<img src="https://github.com/loynin/05-Vehicle-Detection/blob/master/output_images/Original_Lane_Vehicle_box.png" width="800">



---

### Video Implementation

###### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a <a href="https://github.com/loynin/SDCN-05-Vehicle-Detection/blob/master/project_result.mp4"> [link to my video result] </a>


###### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To complish the vehicle detection, I use three sliding windiws (size: 128x128, 96x96, and 80x80) because these windows detect most of the vehicles that they sould have on the image. In block code [11] of notebook, the ```get_hot_boxes()``` is used to detect the vehicles in the image and return all the positive detection.

For overlap detection, I used method of averaging boxes. This method is implemented on the class ```AverageHotBox```. The process of averaging boxes method is to calculate average size of overlaping boxes and create a single new and clean box from the overlaping boxes.

The whole process of the pipeline is in the function ```process_image()```. 

```
def process_image (image_orig):
    
    image_orig = np.copy (image_orig)
    image = image_orig.astype(np.float32)/255
    image_drawed = draw_lane_mark(image_orig)
    # accumulating hot boxes over 10 last frames
    hot_boxes, image_with_hot_boxes = get_hot_boxes (image)
    last_hot_boxes.put_hot_boxes (hot_boxes)
    hot_boxes = last_hot_boxes.get_hot_boxes ()
    
    # calculating average boxes and use strong ones
    # need to tune strength on particular classifer
    avg_boxes = calc_average_boxes (hot_boxes, 10)
    image_with_boxes = draw_boxes(image_drawed, avg_boxes, color=(255, 0,0 ), thick=4)

    return image_with_boxes
```

- Convert image by using ```image = image_orig.astype(np.float32)/255```
- Draw a lane mark on the image (from project 4) ```image_drawed = draw_lane_mark(image_orig)```
- Find the boxes by using ```hot_boxes, image_with_hot_boxes = get_hot_boxes (image)```
- Store boxes in to last_hot_boxes until the limited numbers of frame reach ```last_hot_boxes.put_hot_boxes (hot_boxes)```
- Retreived boxes for calcultion ```hot_boxes = last_hot_boxes.get_hot_boxes ()```
- Find average boxes for vehicles ```avg_boxes = calc_average_boxes (hot_boxes, 10)```
- Draw boxes back to image ```image_with_boxes = draw_boxes(image_drawed, avg_boxes, color=(255, 0,0 ), thick=4)```

Finally, the video processing is in ```process_video()``` function. This function take an input video 'project_video.mp4' and output 'project_result.mp4'.

---

### Discussion

###### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I used have some advantages and disadvantages. An important advantage is speed while I skip the heat map and image threshold it make the process faster. On the other hand, this approach may failed if there are too many cars on the images and some lightning condition that model could not predict accurately. In the future, I will considering to add heat map and threshold to make the process more robus.

Credits:
- Some of code I took from: https://github.com/parilo/carnd-vehicle-detection-and-tracking
- Some of code I took from udacity lesson

