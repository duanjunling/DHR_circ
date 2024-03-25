#DHR_circ
This tool is an animal circRNA-RBP prediction tool. We have developed a deep learning based method to identify RBP binding sites on circRNA. Finally, our training model achieved an accuracy of over 90% on the test dataset. We have developed it into a predictive tool. You can easily use our trained model to perform prediction tasks directly; Alternatively, you can use your own data to train a new model.

#Dependency
python=3.7.7 biopython>=1.76 pysam>=0.15.3 pandas>=1.2.5 numpy>=1.21.6 sklearn>=0.23.1 pytorch>=1.5 bedtools>=2.29.2 CD-HIT>=4.8.1

#Usage
##1. Preprocessing
Whether your data is sequencing data or known circRNA-RBP binding site data, you can use the following steps for preprocessing. Firstly, you need to fill in the path of the ".bed" file in your hand into "positive.py", and you can directly run the "positive.py" script; Next, you need to fill in the path of the "positive.fa" file obtained in the previous step into "negative.py", and you can directly run the "negative.py" script. The positive and negative sample files obtained are "positive.fa" and "negative.fa", respectively.
'''
python positive.py
python negative.py
'''

##2. Prediction
If you need to predict the data in your hands, you can proceed with this step directly after preprocessing. Use our trained model to directly predict and write the score of the results into a file you name yourself.In this step, you can use the following command to obtain the usage method.
'''
python predict.py -h
'''

##3. Training
If you want to use your own data to train the model, you can choose this step after preprocessing. Use 'python train.py -h' to obtain more hyperparameter adjustment instructions.
'''
python train.py -pos_fa (positive.fa) -neg_fa (negative.fa) -out_dir (dirname)
'''

(End)
