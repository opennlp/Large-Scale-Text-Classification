# Large-Scale-Text-Classification

<b> Sparse Victory - A Large Scale Systematic Comparison of count-based and prediction-based vectorizers for text classification </b>
<i> Rupak Chakraborty, Ashima Elhence, Kapil Arora , Proceedings of the Recent Advances in Natural Language Processing, Varna, Bulgaria, 2019 </i> [paper link](https://drive.google.com/file/d/1-8kU0-IPyEV12wRdGaOXxgwM1-GZpJki/view?usp=sharing)
<br>
## Overview
In this paper we study the performance of several text vectorization algorithms on a diverse collection of <b> 73 </b> publicly available datasets. Traditional sparse vectorizers like Tf-Idf and Feature Hashing have been systematically compared with the latest state of the art neural word embeddings like <b><i> Word2Vec, GloVe, FastText </b></i> and character embeddings like <b><i> ELMo, Flair.</b></i> We have carried out an extensive analysis of the performance of these vectorizers across different dimensions like classification metrics (.i.e. precision, recall, accuracy), dataset-size, and imbalanced data (in terms of the distribution of the number of class labels). 
Our experiments reveal that the sparse vectorizers beat the neural word and character embedding models on <b><i> 61 of the 73 </b></i>datasets by an average margin of <b><i> 3-5% </i></b>(in terms of macro f1 score) and this performance is consistent across the different dimensions of comparison.

## Resources
- Datasets used in the experiment can be downloaded from the following [link](http://tinyurl.com/yyofx77r) 
- Pre-trained embedding models can be downloaded from [here](https://tinyurl.com/y2mlnhdf)
- All result files can be viewed [here](https://tinyurl.com/y5e4hftt)
- Detailed visualization of the feature vectors can be seen [here](https://tinyurl.com/yxgf2vuj)

## Steps to execute the code

1. git clone the repository to your local system
2. Run the following command to install all dependencies - 
```markdown
pip install -r requirements.txt
```
3. Download the pre-trained models and create a folder named models in the root directory of the project and put these pre-trained models there
4. Download the datasets from the url provided, then add this path to the file ``` commonconstants.py ``` under the constants package. Also modify other file locations as per your local system requirements
5. Keep a local ``` mongodb ``` instance running to store all the result json files.
6. Run the file ```benchmark_pipeline.py``` under the pipeline package to see the results on the screen.

## Experimental Results

Category Name|GloVe|FastText|Word2Vec|ELMo|Tf-Idf|FeatureHash|Flair
-------------|------|-------|--------|----|------|----------|-------
Sentiment (10)	|41.6/38.1/59.5|	42.9/38.9/59.9|	42.9/38.2/59.4|	36.1/35.1/57.1|	47.0/42.2/63.3|	45.0/41.3/61.8|	43.3/38.9/60.0
Emotion (1)|	14.3/10.3/21.2	|12.5/9.1/20.4	|11.7/9.6/20.8|	7.9/7.0/19.0|	14.2/10.2/19.1|	15.0/10.6/18.3|	8.6/8.2/18.6
General Classification (8)	|56.8/49.5/64.8	|55.9/49.2/64.6	|54.3/48.6/64.0	|46.8/44.9/61.5	|60.7/55.3/68.3	|58.2/51.8/65.1	|56.5/52.2/65.0
Other (5)	|59.7/56.8/67.8	|59.7/56.4/67.4	|59.1/56.6/67.6	|52.9/52.1/65.5	|61.5/55.6/69.8	|57.1/53.3/68.6	|59.1/52.8/67.0
Reviews (2)	|52.1/37.6/83.4	|44.2/37.5/83.2	|52.1/37.6/83.2	|45.6/37.7/83.1	|57.4/43.9/85.4	|50.0/43.6/84.1	|55.8/42.2/84.0
Spam-Fake-Ironic-Hate (5)	|75.9/71.0/82.6	|78.0/72.4/83.7	|77.8/72.4/83.6	|70.7/64.8/81.0	|84.3/79.3/87.6	|80.0/74.9/84.5	|79.9/76.3/85.4
Medical (4)	|45.2/40.2/70.3	|42.9/40.3/70.1	|45.6/40.8/70.3	|40.6/36.9/68.7	|53.8/45.9/73.8	|47.3/42.2/70.6	|49.3/42.2/71.3
News (4)	|50.6/49.4/66.6	|48.6/48.3/66.2	|48.9/48.7/66.1	|35.9/36.6/54.3	|63.0/60.0/77.6	|58.1/55.8/73.2	|63.2/60.9/78.4

The table presents the values for <b> Precision/Recall/Accuracy </b>, the results have been averaged across all the classifiers used in the study. The size of the datasets used in the table is less than or equal to 10K. Please refer to our paper for detailed results over the entire dataset.

<p float="left">
<img src="https://i.ibb.co/F81m57Y/accuracy-vectorizer.png" width="150" />
<img src="https://i.ibb.co/cYJC268/classifier-accuracy.png" width="150" />
<img src="https://i.ibb.co/56KWhGC/f1-classifier.png" width="150" />
<img src="https://i.ibb.co/5vCF91F/f1-vectorizer.png" width="150" />                                             
</p>                                            

The images given above show the following metrics (from left to right) - 1. Violin Plot showing the accuracy of all the vectorizers used in the study across all the datasets. 2. Violin Plot showing the accuracy of the classifiers used in the present study, under the same conditions as 1. 3. Macro f1-score of the classifiers used. 4. Macro f1-score of the vectorizers used.

### Support or Contact

We are always happy to receive feedback on ways to improve the framework. Feel free to raise a PR in case of you find a bug or would like to improve a feature. In case of any queries please feel free to reach out to [Rupak](mailto:rupak97.4@gmail.com)
