## Using Text Classification in Recurrent Neural Networks to Detect Fake News

**Author: Melissa Paciepnik**

### Executive summary
**Project Overview and Goals**

The goal of this project is to create a model that accurately classifies news as Real or Fake based on its linguistic contents.  The motivation behind this is to review the effectiveness of machine learning models at classifying high stakes information such as news.  In a world that is irreversibly entrenched in social media and AI tools, it is crucial to build trust in verified news sources and stem the spread  of misinformation.  In reality we know that this is a multifacted effort that requires media and societal integrity, regulation policies, and general education.  However it is equally important to both find ways to create reliable tools for flagging fake news, as well as educating the public on the limitations of such methods, since there is a high societal cost to classifying both fake and real news incorrectly. 

**Findings**

The best model for correctly classifying news was a recurrent neural network (RNN) model, with an accuracy of 0.856, F1 score of 0.855, recall of 0.922, precision of 0.797 and PR-AUC of 0.959 on unseen test data.  This was trained on a dataset of 750 text samples that were classified as 0 (Real) or 1 (Fake).  The scores mean that model on the 250 new test samples, the model correctly classified 85.6% of samples, with 27 False Positives (Real news incorrectly identified as Fake) and 9 False Negatives (Fake news incorrectly identified as Real). Note that due to the inherent nature of neural networks, the results in the jupyter notebook change slightly each time it is run, however the RNN model still far outperforms the other models each time.

### Research Question
The goal of this project is to determine the best model for predicting whether or not news articles are from a fake or verified sources based on the linguistic contents of the article.

### Rationale
The importance of my project has two primary parts:

1. It continues the discussion on building trust in society through developing and providing effective tools to be able to critically analyze the validity of the extreme amount of information being shared online. 
    
	- This theoretically could help accurately inform public opinions and sentiments in times of massive misinformation spread when major world events are occurring, or even during minor public relation events. As one explicit example, this model could be used by social media companies to highlight what they consider as verified information, and flag others as needing more validation or analysis. In another context, public relations teams at companies could train a similar model to recognize their publically available statements, and be used to flag bogus articles that are publishing incorrect statements about their brands  
	- This is of critical importance in today's society where fake or unverified news proliferates like rapid fire through social media, and is extremely influential on public opinion and actions.

2. The other more philosophical benefit of my project is to be able to communicate to users what the reliability and limitations of this type of modelling is. 

	- In reality, we know that large language models are at the forefront of AI research and products, and that there are much more powerful models that are trained on enormous databases and with far greater processing capacity than what was used for this project, that could be trained to detect real or fake news based on their contents with greater success in a wider range of contexts.
    - The importance of conducting this type of more simplified modelling lies in increasing general public understanding of how ML/AI works with regards to these kinds of truth seeking activities. This points to the broader and more important question that faces society on improving the actual reliability of classification models - the truth depends largely on the accuracy and integrity of the human inputs to classifying what is true or false in the data that the machine is learning from.
    - It is important and humbling to acknowledge that the scope of my capstone also depends on the accuracy of the dataset I'm using in classifying news articles as 'fake' or 'real'.  Already reading the comments on the dataset, it appears that the author of the kaggle dataset incorrectly described the classifiers as their opposite - which already presents a problem that needs cleaning, as we will do within the notebook.
    - The intent is to help educate the public and make data science and activities more transparent, which is crucial in an age where people put so much faith into what they see online.  It entails breaking down the expectations of what information ML/AI can give us, as well as holding both corporations and individuals accountable for the type of ML/AI that they are spreading in the world.  This allows the public to stand up to false information and false methods for checking information

### Data Sources
The dataset used in this report was sourced from Kaggle at the following [link](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data).

### Methodology
The analysis was conducted as follows:

1. **Data cleaning and pre-processing**

	- a) Removal of unnecessary columns, duplicate rows, missing data
	- b) Processing raw text from the title and text of each article with the aim of later converting the text into numerical data that can be processed by machine learning algorithms. This included:
 		- i) Tokenizing text which splits the raw text into a vector of separate words and punctuation.
    		- ii) Removal of common stop words and punctuation.
      	 	- iii) Lemmatization of words (reducing words down to their base/root stems to remove some noise).
   	- c) Creating a clean final dataset to be processed by machine learning techniques consisting of:
   		- i) X matrix: contained 2 columns, the the post-processed 'Title' and 'Text' entries  for each article.  Only text or title was used for modelling due to processing constraints, but both were processed to allow flexibility for testing each of the models.
   	 	- ii) y matrix: contained the classification for each article (0 - Real, 1 - Fake).       
	- d)  Splitting the data into train and testing sets using a 75/25 default split.
		- This is done to allow preliminary cross-validation and evaluation of a model.  The model is trained on the training dataset, and then fit to the previously unseen test data.  If the model is not overfit and performs well on unseen data, it is more likely to be able to generalize to new unseen data.

3. **Exploratory data analysis**

	- a) Visualizing distributions of word and character counts for title and text.
 	- b) Summarizing and visualizing most frequently occuring words for each class (Fake, Real).
	- c) Visualizing the distribution of class in the data.
 		- i) The dataset was slightly imbalanced, with an approximate 55/45% split between Real/Fake news.
   	- d) Calculation of a baseline model
		- i) Given that this is a binary classification where the task is to choose between two classes, the baseline model is set to classifying everything as the Majority Class.  This means that the model guesses everything is Real news, which gives 55% accuracy on our existing dataset.  
		- ii) Our machine learning models should have better accuracy than this baseline.

5. **Selecting best metrics to evaluate best performing model**
   
    Since our model is slightly imbalanced, we cannot rely only on high accuracy as a metric of model success. 
    In our dataset, the 'positive' case (1) is 'Fake' news, and 'negative' case (0) is 'Real' news. 
    In this context, there is a high societal cost to both:

     - a) False positives - incorrectly identifying real news as fake (undermines legitimate news sources and spreads distrust).
     - b) False negatives - incorrectly identifying fake news as real (further spreads misinformation and unnecessary hysteria).
    
    As such, the F1 score was chosen as the best indicator of model performance:
    
	- The F1 score is the harmonic mean between precision and recall:
    		- It is calculated as 2 x (Precision x Recall) / (Precision + Recall)
		 
    - Precision is:  	 
	    - "Of all the samples that were labelled as Fake news, how many were actually Fake?"
        	- It is calculated as (True Positives (TP)) / (True Positives (TP) + False Positives (FP))
          	- If there is a high False Positive rate (Real news incorrectly labelled as Fake), precision will be low.
       
    - Recall is:  	 
	    - "Of all actual Fake news in the sample, how many did the model correctly identify?"
        	- It is calculated as (True Positives (TP)) / (True Positives (TP) + False Negatives (FN))
          	- If there is a high False Negative rate (Fake news incorrectly labelled as Real), recall will be low.

    The F1 score can only be high if there is high precision and high recall.
   	- Thus it is a good metric to maximize when the cost of False Negative and False Positives is high, which is true for our problem.
   	- In our case, we consider the best performing model to be the one that has an **F1 score as close as possible to 1**, as this minimizes the risk of both False Positives and False Negatives. 

#### Results
This model performed the best compared to the non neural models for correctly classifying news as Fake or Real.  Note that due to the inherent nature of neural networks, the results in the jupyter notebook change slightly each time it is run, however the RNN model still far outperforms the other models each time.

During this run, the best model was the RNN with an F1 score of 0.855.
Its optimal hyperparameters were 10 epochs and 32 neurons. 
On the test data, it achieved 0.856 accuracy, 0.922 recall, 0.797 precision and 0.959 PR-AUC.

Of the non-neural models, Logistic Regression performed the best with an F1 score of 0.775.

Below is a summary of how each F1 score affected the actual False Positives (Real news incorrectly identified as fake) and False Negatives (Fake news incorrectly identified as Real) by each model out of the test dataset of 250 samples:

- RNN model
    - F1 Score: 0.855
    - False Positives: 27
    - False Negatives: 9
    - Total incorrect classifications: 36 (14.4%)

- Logistic Regression model
    - F1 Score: 0.775
    - False Positives: 24
    - False Negatives: 27
    - Total incorrect classifications: 51 (20.4%)
    
- SVC model
    - F1 Score: 0.730
    - False Positives: 12
    - False Negatives: 42
    - Total incorrect classifications: 54 (21.6%)
    
- Naive Bayes model
    - F1 Score: 0.677
    - False Positives: 16
    - False Negatives: 48
    - Total incorrect classifications: 64 (25.6%)
    
- Decision Tree model
    - F1 Score: 0.490
    - False Positives: 2
    - False Negatives: 77
    - Total incorrect classifications: 79 (31.6%)

### Next steps
1. Improve model performance with further tuning of parameters in the RNN model such as additional layers, neuron counts, batch size, number of epochs.
2. Use pre-trained language models such as BERT to process and train the data. 

### Outline of project

- [Link to Initial Technical Report](https://github.com/mpacielim/FakeNewsCapstoneInitialReport/blob/main/Fake%20News%20Capstone%20Initial%20Technical%20Analysis.ipynb)
- [Link to Raw Fake News Dataset](https://github.com/mpacielim/FakeNewsCapstoneInitialReport/tree/main/data)
- [Link to Raw Fake News Dataset Original Location in Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data)
- Note that the dataset linked above was above the maximum 100 MB file size upload allowed by github, hence the [github large file storage (LFS) system](https://git-lfs.com/) was used.
	- The .gitattributes file in this repository was required to upload the dataset through the github LFS platform.


##### Contact and Further Information
Melissa Paciepnik
[LinkedIn](https://www.linkedin.com/in/melissa-paciepnik-43a85979/)
