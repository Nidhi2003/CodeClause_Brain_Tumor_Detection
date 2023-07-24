# CodeClause_Brain_Tumor_Detection
This is one of my task of the internship by CodeClause.
According to the International Agency for Research on Cancer (IARC), the mortality rate due to brain tumors is 76%. 
It is required to detect the brain tumors as early as possible and to provide the patient with the required treatment to avoid any fatal situation.
With the recent advancement in technology, it is possible to automatically detect the tumor from images such as Magnetic Resonance Iimaging (MRI) 
and computed tomography scans using a computer-aided design. 
Machine learning and deep learning techniques have gained significance among researchers in medical fields, especially Convolutional Neural Networks (CNN),
due to their ability to analyze large amounts of complex image data and perform classification.
The objective of this review article is to present an exhaustive study of techniques such as preprocessing, machine learning, and deep learning
that have been adopted in the last 15 years and based on it to present a detailed comparative analysis.
The challenges encountered by researchers in the past for tumor detection have been discussed along with the future scopes that can be taken by the researchers
as the future work. Clinical challenges that are encountered have also been discussed, which are missing in existing review articles.

The proposed model is evaluated on different parameters. These parameters are carried out to determine whether the proposed model is better than the previous methods and whether it is appropriate for brain tumor detection or not. The proposed model is implemented on an MRI brain image dataset, and a deep learning model is applied to this dataset.

# Evaluation metrics
Different evaluation metrics are used for the prediction and classification challenges, such as accuracy, precision, recall, and F1-measure. The effectiveness of the proposed model is assessed using the following evaluation metrics.

Accuracy: To measure the accuracy of the proposed model, compute the ratio of the false positive, true positive, true negative, and false negative. The Equation (1) represent the accuracy estimate.

Accuracy=TP+TNTP+TN+FP+FN    (1)
Precision: The ratio of real positives to all positives overall (both false and true) in the data. Additionally referred to as a high predicted value. The precision rate is represented in Equation (2).

Precision=TPTP+FP    (2)
Recall: The ratio of the true positives in the data to true positives and false negatives is often referred to as sensitivity, the chance of detection, and the rate of a true positive. The recall rate is represented in Equation (3).
Recall=TPTN+FN    (3)
 
F1-measure: The weighted average of the precision and recall is known as the F1 measure. The Equation (10) represents the value of F1-measure.

F1−measure=2×Precision+RecallPrecision+Recall    (4)

# Experimental analysis and result
The proposed technique is applied to deep learning models CNN and CNN-LSTM. Figure 7 represents the performance evaluation of the convolutional neural network with training and validation loss and accuracy. At every 100 epochs, the training accuracy is 99.4%, and validation accuracy is 98.3%. Similarly, the training loss is 0.007, and 0.113 is the validation loss.
