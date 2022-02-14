# ML-7641
REPO for CS-7641



Author: Dustin Ingersoll (dingersoll3)

Env: PyCharm 2021.3, python 3.8

Date: 02/13/2022


=== SUMMARY ===

Code link: https://github.com/dustin-ingersoll/ML-7641

Included is the source code, datasets, and graphs used in the analysis.


The raw datasets can be found here (resulting filenames need changing to below names to run):

Spam: https://www.kaggle.com/chandramoulinaidu/spam-classification-for-basic-nlp

Phishing: https://www.kaggle.com/shashwatwork/web-page-phishing-detection-dataset


MLA1
|
|- data    ( spam and phishing raw datasets used )
|
|---- dataset_phishing.csv    ( phishing dataset )
|
|---- dataset_spam.csv     ( spam dataset )
|
|- graphs     ( the resulting graphs used in the report )
|
|- boosting.py     ( test code for boosting learner )
|
|- decision_trees.py     ( test code for DT learners )
|
|- knn.py     ( test code for KNN learners )
|
|- neural_networks.py     ( test code for NN learners )
|
|- preprocessing.py     ( code for cleaning X and Y data, as well as test functions )
|
|- requirements.txt      ( required imports to run project in python environment )
|
|- run.py      ( functions to run all tests consecutively )
|
|- svm.py      ( test code for SVM learners )



=== USABILITY ===


1) Install the requirements.txt into your python environment
2) Run the file "run.py". This will run all tests and replace the graphs currently in the project.

** NOTE: running this file will take a LONG time and requires a lot of processing to complete as it does each test 10 times. If you would like to cut down on this time, edit all existing tests in the learner files to include "rounds=1" in all test_attributes() calls. 



=== EDITING ===


The main function needed to alter / run your own tests is preprocessing.test_attribute().

Models and attributes mentioned are found in the Scikit-learn library.

This is the main function which acts as a pipeline to automated testing. It takes:

 - X and Y data
 - a Model (learner model)
 - attribute to test
 - number of rounds to randomize, test, then average over
 - a range or dictionary to iterate over the attribute
 - additional attributes to include in the learner in addition to the target attribute
 - x and y labels for the resulting graph
 - title of the resulting graph
 - filename to save the graph, otherwise just shows the plot without saving.

All tests performed in this analysis use this function to automate the data splitting, learner creation, testing, and graphing. If you would like to edit an existing test or create your own, follow the existing tests within the learner files as a template. 

** note: tests were NOT run using randomization seeds. Every run will be slightly different. 

** note: this function is NOT thoroughly tested and may not perform up to production requirements. 

