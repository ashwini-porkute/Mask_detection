# Mask_detection
Date: 20 October 2020
Folder: /home/ashwini/FaceMaskDetectionProject

Step 1 : Face Mask Dataset Preparation:
=========================================================================
Collect the face mask dataset from Kaggle (account: ashwiniprkt@gmail.com, pswd: ashpor189)
https://www.kaggle.com/andrewmvd/face-mask-detection

Step 2: Dataset Preprocessing:
=========================================================================
the Kaggle_FaceMask_dataset folder structure is as follows:
a) Images
b) Annotations

Images folder will have all .png format images whereas in annotations folder we will find the annotations of respective images in xml format.
So, as here we will be using the Tensorflow's Object detection API, we will need the datset in TFRecord format.
So our primary focus will be on converting/processing our datset to TFRecord format and for that we need to convert all .xml files to .csv.

HOW TO PARSE XML FILES SEE: https://www.tutorialspoint.com/python/python_xml_processing.htm
