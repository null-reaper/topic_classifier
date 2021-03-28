# topic_classifier
A naive bayes classifier trained on speeches given by democratic/republic politicians

The dataset consists of 2 text files (train.txt and test.txt) each containing a label RED (for republican) or 
BLUE (for democratic) followed by a tab character and the corresponding speech on each line. The training
dataset contains 46 such examples, while there are 18 in the test set—the labels are evenly distributed.

The naive bayes classifier estimates the probability p(label|words) by first computing p(label) and p(words|label);
probability calculations are performed in log space and add-one smoothing is applied to improve results.

The output when evaluating the classifier on the test dataset were as follows: <br />

overall accuracy: 0.9444444444444444 <br />
precision for red: 0.8571428571428571 <br />
precision for blue: 1.0 <br />
recall for red: 1.0 <br />
recall for blue: 0.9166666666666666

