##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image1]: ./output_for_readme/raw-vehicles-GTI_MiddleClose-image0425.png "raw car image"
[image2]: ./output_for_readme/raw-non-vehicles-Extras-extra2126.png "raw not-car image"
[image3]: ./output_for_readme/HOG-vehicles-GTI_MiddleClose-image0425.png "car HOG image"
[image4]: ./output_for_readme/HOG-non-vehicles-Extras-extra2126.png "not-car HOG image"
[image5]: ./output_for_readme/bounding_boxes_test1.jpg "bounding boxes test_images/test1.jpg"
[image6]: ./output_for_readme/bounding_boxes_test2.jpg "bounding boxes test_images/test2.jpg"
[image7]: ./output_for_readme/bounding_boxes_test3.jpg "bounding boxes test_images/test3.jpg"
[image8]: ./output_for_readme/bounding_boxes_test4.jpg "bounding boxes test_images/test4.jpg"
[image9]: ./output_for_readme/bounding_boxes_test5.jpg "bounding boxes test_images/test5.jpg"
[image10]: ./output_for_readme/bounding_boxes_test6.jpg "bounding boxes test_images/test6.jpg"
[image11]: ./output_for_readme/single_img_label_test6.jpg "single img labels test_images/test6.jpg"
[image12]: ./output_for_readme/single_img_heatmapped_test6.jpg "single img heatmap test_images/test6.jpg"
[image13]: ./output_for_readme/img_detection_test1.jpg "img detection test_images/test1.jpg"
[image14]: ./output_for_readme/img_detection_test2.jpg "img detection test_images/test2.jpg"
[image15]: ./output_for_readme/img_detection_test3.jpg "img detection test_images/test3.jpg"
[image16]: ./output_for_readme/img_detection_test4.jpg "img detection test_images/test4.jpg"
[image17]: ./output_for_readme/img_detection_test5.jpg "img detection test_images/test5.jpg"
[image18]: ./output_for_readme/img_detection_test6.jpg "img detection test_images/test6.jpg"
[image19]: ./output_for_readme/resulting_bounding_box_test1.jpg "resulting bounds test_images/test1.jpg"
[image20]: ./output_for_readme/resulting_bounding_box_test2.jpg "resulting bounds test_images/test2.jpg"
[image21]: ./output_for_readme/resulting_bounding_box_test3.jpg "resulting bounds test_images/test3.jpg"
[image22]: ./output_for_readme/resulting_bounding_box_test4.jpg "resulting bounds test_images/test4.jpg"
[image23]: ./output_for_readme/resulting_bounding_box_test5.jpg "resulting bounds test_images/test5.jpg"
[image24]: ./output_for_readme/resulting_bounding_box_test6.jpg "resulting bounds test_images/test6.jpg"


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This is the required README.  
All code cells mentioned here are in the IPython notebook located in "vehicle_detection.ipynb".  
For brevity, i will just refer to the IPython notebook as simply 'notebook'.  
###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cells 3 through 7 of the  notebook.

Code cell 3 - simply displays some random car and not-car image.  
![alt text][image1]  
![alt text][image2]  
Code cell 4 sets up some tentative HOG and other related parameters.  
Code cell 5 establishes a set of helper functions for feature extraction. First get_hog_features() is setup as a helper function that internally wraps  skimage.feature.hog() which in turns relies on the HOG parameters established in cell 4. Subsequently extract_features() internally calls get_hog_features() and depending on whether only a specific channel is passed in or 'ALL' is passed in, returns appropriate features.  
Code cell 6 displays the HOG images of the same random car and not-car images shown earlier.  
![alt text][image3]  
![alt text][image4]   
Code cell 7 extracts the car and not-car features from the data and makes those globally available to the rest of the notebook.  

####2. Explain how you settled on your final choice of HOG parameters.

Its hard to tell just from the HOG extracted features that they will be useful downstream in actually being able to position the bounding boxes correctly on the region of interest , within a given image. So i settled on some values provided in the lectures and some by back tracking from when an approximate good quality bounding box image was found on the a few of the sample images.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Code in cell 8 trains the classifer. We first preprocess the data to apply a sklearn.preprocessing.StandardScaler() , then we do a 80/20 train/test random selection, then apply sklearn.svm.LinearSVC() to the data, and finally store the resulting classifer ina pickle.   

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in the code cells 9 and 10 of the  notebook.

Code in cell 9 setups several helper functions. The most important of these is the search_windows(0 function. This function takes an image, a list of window objects, a classifer and sevarl other parameters, and returns an array of positive detection windows. the list of window objects parameter is typically supplied by another helper function , slide_window(). This second function takes an imge, and certain bounds such as x_start/stop and xy_overlap etc to return an array of windows covering the region of interest in the single image provided. Scales and overlap were selected with some trial and error and also based on some slack channel and class viedos.  
Code in cell 10, essentailly does first pass run of the earlier tentative parameters on a single image.  

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 'ALL' channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]  
![alt text][image6]  
![alt text][image7]  
![alt text][image8]  
![alt text][image9]  
![alt text][image10]  


### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./movie.mov)    
Here's the corresponding youtube link - https://www.youtube.com/watch?v=BLQsqb4OVlQ   
Normally this would have been an mp4 file. However i found that running the 1261 or so frames of the video on an AWS gpu instance takes upwards of 2 hour 30 minutes or so. So i use VideoClip.subclip() to split into four clips of roughly equl size and run four AWS instances in parallel, taking care to use the same classifer.pickle file in each instance. This eventually did have some benefit as the single notebook normally took about 7.3 seconds to process each frame, and with four notebooks running, the time for overall run was about half hour. Subsequently, when the four processed mp4 subclips were downloaed to local mac, and then joined together using quicktime, the resulting video can only be saved as .mov file.

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in the code cells 16 through 18 of the  notebook.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from test_images/test6.jpg:
![alt text][image11]  
![alt text][image12]  

####3. A method, such as requiring that a detection be found at or near the same position in several subsequent frames, (could be a heat map showing the location of repeat detections) is implemented as a means of rejecting false positives, and this demonstrably reduces the number of false positives. Same or similar method used to draw bounding boxes (or circles, cubes, etc.) around high-confidence detections where multiple overlapping detections occur.

The code for this is cell 18, and its wiring into the video pipleline is in cell 23.

As advised i have uses a deque with maxlen=5, to store the average of the last preceeding 5 heatmaps. In the early frame, where the frames < 5 have not yet been processed, the average is the sum of whatever frames we have processed so far.

### Here are six frames and their corresponding heatmaps:
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]


### Here the resulting bounding boxes are drawn onto the same six frame in the series:
![alt text][image19]
![alt text][image20]
![alt text][image21]
![alt text][image22]
![alt text][image23]
![alt text][image24]


###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Rerunning/reprocessing the entire video took some creativity. Notebooks would routinely timeout while processing. I learned that they notebooks can be run with a parameter to override the default 60 seconds timeout. Further the subclip() method worked well for quickly iterating over small clips, and then only running the final 4-split subclips one time.
In addition, just as an iterative proecess , i dropped the overalp to 60% from the original 75%. This appears to have led to a situation where , especially for the white car, there briefly appears two bounded rectangles over the same car. Perhaps with a greater overlap, i could have avoided it.
