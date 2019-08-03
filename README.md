# Large-Scale-Text-Classification

<b> Sparse Victory - A Large Scale Systematic Comparison of count-based and prediction-based vectorizers for text classification </b>
<i> Rupak Chakraborty, Ashima Elhence, Kapil Arora , Proceedings of the Recent Advances in Natural Language Processing, Varna, Bulgaria, 2019 </i>
<br>
## Overview
In this paper we study the performance of several text vectorization algorithms on a diverse collection of <b> 73 </b> publicly available datasets. Traditional sparse vectorizers like Tf-Idf and Feature Hashing have been systematically compared with the latest state of the art neural word embeddings like <b><i> Word2Vec, GloVe, FastText </b></i> and character embeddings like <b><i> ELMo, Flair.</b></i> We have carried out an extensive analysis of the performance of these vectorizers across different dimensions like classification metrics (.i.e. precision, recall, accuracy), dataset-size, and imbalanced data (in terms of the distribution of the number of class labels). 
Our experiments reveal that the sparse vectorizers beat the neural word and character embedding models on <b><i> 61 of the 73 </b></i>datasets by an average margin of <b><i> 3-5% </i></b>(in terms of macro f1 score) and this performance is consistent across the different dimensions of comparison.

