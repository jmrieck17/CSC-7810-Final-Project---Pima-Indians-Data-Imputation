# CSC 7810 Final Project - Pima Indians Diabetes

## A Study of Imputation Methods and its Impact on Machine Learning Algorithm Performance

![image](https://user-images.githubusercontent.com/75294739/231504426-6767683c-05c0-4cb3-b20c-53dc3356785e.png)

# Abstract
This research paper aims to test different imputation methods in filling in missing values using the Pima Indians diabetes dataset, and to evaluate the performance of several machine learning models in predicting diabetes outcomes using the imputed dataset. The dataset contains medical information on Pima Indian women, including several features such as age, body mass index (BMI), and glucose levels. The missing values in the dataset were imputed using different methods, including k-nearest neighbor imputation, linear regression imputation, and random forest imputation. Each imputed dataset was then trained on eight different machine learning models, with cross validation used on each algorithm to find the optimal set of hyper-parameters. The performance of these models was evaluated using area under the curve (AUC) scoring. The study found that imputation methods can significantly impact feature distribution and model performance, with some methods resulting in improved performance and others leading to decreased performance. The KNN model consistently showed improvement across all imputation methods, while the decision tree model had the poorest performance. The best overall performance was achieved by the Keras artificial neural network using random forest imputation, with glucose identified as the most important variable for predicting the diagnosis. The results suggest that careful consideration of imputation methods and learning models is crucial for accurate and effective analysis of data with missing values.

# Introduction
When training machine learning models, missing data can create significant challenges in accurately predicting outcomes. As the adage goes, "garbage in, garbage out," emphasizing that the quality of data inputs will dictate the quality of model outputs. Missing data can not only impact the performance evaluation of machine learning models but also result in issues such as overfitting and inaccurate prediction results on newer data. 

In the past, two techniques commonly employed to address missing data are deletion of instances with missing values and auto-filling with the mean score within a feature. While these two techniques have offered a quick and simple solution to dealing with missing values, they can have their limitations. For instance, deletion can potentially lead to significant data loss, while filling missing values with the mean can introduce more bias in the dataset, which can skew the feature data distribution, resulting in inaccurate results. However, using machine learning imputation methods can help preserve the data's underlying structure, reduce bias, and make better feature predictions by considering the relationships between all the features in the dataset. 

## Study Objectives
The objective of this paper is to implement and observe several machine learning imputation methods and test to see if there is any noticeable performance improvement across multiple machine learning algorithms. Each machine learning algorithm will have their hyper-parameters fine-tuned in order to ensure the highest performance result. 

## Historical Review
According to the World Health Organization, diabetes is a chronic disease that affects the body’s ability to produce and regulate insulin levels within the body that control an individual’s blood glucose (World Health Organization, 2022). Long term diabetes not only affects the pancreas, which produces the insulin hormone, but can also lead to co-morbidities including cardiovascular disease, kidney disease, and nerve damage in limbs and extremities. Complications from diabetes can also lead to amputation and blindness in severe cases (World Health Organization, 2022). This is why early detection is so important for diabetes patients. Early detection allows for healthcare professionals to quickly intervene and put patients on treatment plans that help curb dangerous co-morbidities above and improve quality of life.  

The Pima Indians are a Native American tribe located in the Southwest United States, where they have 52,600 acres of tribal lands just outside of Phoenix, AZ (Inter Tribal Council of Arizona, 2023). Nearly 6,000 years ago, the Pima tribe's ancestors settled around the Gila River basin and established an intricate man-made irrigation system that consisted of 500 miles of canals and ditches; with this system, they successfully cultivated the land, supporting a population that reached 50,000 to 60,000 people at its peak (Inter Tribal Council of Arizona, 2023). Early Spanish settlers made contact with the tribe and established trade of new crops and livestock, then in 1846, the US government established ties with the tribe after the Mexican American War (Inter Tribal Council of Arizona, 2023). 

Although the Pima tribe treated American settlers who ventured West in search of gold with friendliness, the incoming settlers constructed their own dams and irrigation system upstream on the Gila River in the 1870s and 1880s, which unsurprisingly resulted in a severe famine and widespread starvation among the established Pima tribe living downstream (Inter Tribal Council of Arizona, 2023). The transition from locally sourced food to highly processed canned food provided by the US government lead to the development of high rates of obesity and diabetes amongst the tribe (Inter Tribal Council of Arizona, 2023). In fact, according to a comprehensive global study on diabetes rates among regional populations, the Pima Indian tribe had the highest rate of diabetes recorded. The study revealed that 42% of participants aged 25 and older had diabetes, while 50% of participants aged 35 and older had the disease (Bennett, Burch, & Miller, 1971).

The dataset utilized in this project is composed of de-identified health data from 768 individuals, collected by the National Institute of Diabetes and Digestive and Kidney Diseases. The dataset has been widely used in diabetes research, with one of the earliest papers citing it dating back to 1988. For the purposes of this study, the patient selection criteria were women aged 21 or older with Pima ancestry (UCI Machine Learning, 2023). 

## Literature Review
There have been many studies that have been conducted using the Pima Indians diabetes dataset throughout the years. Not only have there been countless studies based on various machine learning models and their performance on accurately predicting a diabetes diagnosis, there have been studies that take into consideration what learning models are the most transparent for healthcare professionals to feel confident in a diagnosis prediction. A study completed in 2022 by Chang, V et al proposed that the best learning models to implement in an Internet of Medical Things (IoMT) environment that give healthcare providers access to the internal decision-making process are decision tree, naïve bayes, and random forest models (Chang, Bailey, Xu, & Sun, 2022). 

Other studies that have used this dataset have explored different feature reduction methods to improve prediction performance. One study done in the Journal of Statistics and Mathematical Engineering used Exploratory Factor Analysis (EFA) to combine various features such as BMI and Skin Thickness, Insulin and glucose, pregnancies and age into three respective factors for improved predictive performance (Biju & James, 2022). Another study looked at the effects of dimensionality reduction on the data using principal component analysis (PCA) and tested the performance accuracy using a naïve bayes learning model, resulting in a less than 1% improvement (Ozsahin, Mustapha, Mubarak, Ameen, & Uzun, 2022). 

There continue to be new studies done by the National Institute of Diabetes and Digestive and Kidney Diseases, located in Phoenix, AZ that has utilized more in-depth patient data collected from the local Pima Indian community. One of these newer studies is exploring how diabetes may lead to an increased risk of cognitive decline and dementia (Biessels & Despa, 2018).

# Methodology

## Defining the Dataset Variables
The Pima Indians dataset has a total of 768 instances and nine numeric feature variables. The table below lists the feature variables that were collected from each individual, a brief description of what each one means, and the range of observed values for each variable:

![image](https://user-images.githubusercontent.com/75294739/231308794-bea573f3-9084-4113-b02c-5ff7813d1c88.png)

Upon closer inspection of the dataset, it becomes apparent that there are several instances where missing data is represented as a value of 0 instead of a blank or null value. To illustrate this point, the graph and table below present the distribution of each feature variable, separating the counts of values greater than zero from those equal to zero:

![image](https://user-images.githubusercontent.com/75294739/231309535-3fbe6434-3773-471a-8a41-ef007c5eef8e.png)
 
Although it is reasonable to anticipate that some individuals may have no pregnancies at the time of evaluation, it is still possible that some of the zero values for this feature may actually be null. Among the feature variables, insulin had the highest number of null values, with 374 out of 768, while Skin Thickness had the second highest number of null values, with 227 out of 768.

Next, we will examine the correlation between each feature variable. This is an essential process that helps to determine whether any of the feature variables are dependent on each other. The independence of features is significant because some of the learning models used in this study, such as Naïve Bayes, assume that all features are independent. Therefore, understanding the correlation between each feature variable is crucial in evaluating and understanding the performance of the machine learning models.

![image](https://user-images.githubusercontent.com/75294739/231309559-219169d4-0635-4493-b771-2e8d23cc8553.png)

Based on the correlation graph above, it can be observed that there is no strong correlation between each individual feature variable or between the feature variables and the output variable As a result, there isn’t an additional data pre-processing requirement to deal with feature dependency in the dataset. 
  
## Defining Imputation Steps
In this experiment, we will compare the performance of three different imputation methods to fill in the missing null values in the dataset: K-Nearest Neighbors Imputer,  Linear Regression, and Random Forest Imputer. Each method uses the respective algorithm to input the null values.

KNN and random forest imputation methods assume that the data is non-parametric, while linear regression relies on the assumption that the variable relationship is linear (Faisal, 2018) (Dash, 2022). Consequently, each algorithm manages missing data differently. Linear regression imputation relies on complete individual cases to predict the missing values (Swalin, 2018), but for this dataset, the linear regression model can only rely on 392 out of 768 total instances. In contrast, both KNN and random forest imputers directly handle each null value by looking at the values of all variables across the dataset, instead of limiting themselves to only complete instances (Swalin, 2018). 

## Data Pre-Processing Steps
After a determination has been made regarding what imputation method (if any) is applied to the dataset for testing, the data will be put through a scaling process using the standard scale. This scaling method places each feature variable’s mean at 0 and sets the standard deviation to 1, while preserving the original distribution of each feature. Standard scaling also works better than other scaling methods when it comes to outlier sensitivity, due to the mean and standard deviation being the same across all feature variables.

The dataset will then be split into two subsets using a train-test ratio of 80-20. This ratio is commonly used in most machine learning studies and allows for the learning algorithms to train on more data. The 80-20 split also decreases performance variance and allows for algorithms to have a more stable output. 

## Defining Machine Learning Algorithms Used
### Logistic Regression
Logistic Regression classifier is a standalone binary classification algorithm that uses feature variables to predict outcomes. The algorithm transforms each feature variable into a probability between 0 and 1 using the sigmoid formula, which acts as an activation function:

Sigmoid=  1/(1+ e^(-x) )

This prediction is then compared what the true classification is using the log-loss function: 

Log-Loss = -y_i ln(f(x_i ))-(1-y)ln(1-f(x_i ))

The algorithm adjusts the weights of the prediction model to minimize the loss and produce accurate predictions. This algorithm works well at predicting binary classifiers (Thorn, 2020).

### Naïve Bayes
Naïve Bayes classifier is another standalone algorithm that uses Bayes Theorem to predict the probability of an outcome (A), given a set of features (B):

P(A│B)=  (P(B│A)*P(A))/(P(B))

Naïve Bayes assumes that each feature is independent from one another, hence it being naive (Gandhi, 2018). One drawback to this assumption of feature independence is how each feature is correlated to one another. Features that have correlation could lead to performance issues with the model.

### Decision Tree
The Decision Tree is another standalone algorithm that uses rule-based logic to create a root node and split the data into various leaf nodes by calculating lowest loss function for each split. Like logistic regression, the decision tree algorithm can use the log-loss function. Other loss functions that can be used are entropy loss:

Entropy= -∑_(i=1)^n▒p_i *log (p_i)

And Gini Index loss:

Gini Index = 1-∑_(i=1)^n▒(p_i )^2 

Decision trees can handle both categorical and numeric feature variables and is easy to visualize and explain. Decision trees are also good at handling datasets where the relationships between each feature are non-linear. The only drawback to decision trees is that they can overfit training data, which can lead to lower performance on the testing data (Bento, 2021) (scikit learn, 2023). Decision trees also run into performance issues when there is an imbalance in the distribution of classifier data, which can cause issues when nodes are being split (Brownlee, 2020).

### K Nearest Neighbor
The K nearest neighbor (KNN) classifier is a standalone algorithm that works with both regression and classification models and assigns a prediction based on both distance and majority vote. There are several different distance metrics that can be used to select the nearest neighbor points for the majority vote. The distance metrics used in this project are the following:

Euclidean Distance = √(∑_(i=1)^n▒〖(y_i  - x_i  )〗^2 )

Manhattan Distance = (∑_(i=1)^n▒|X_i-y_i  | )

Minkowski Distance = 〖(∑_(i=1)^n▒|X_i-y_i  | )〗^(1/p)

Another important feature of the KNN classifier is the hyper-parameter n_neighbors, which is used to determine how many nearest neighbors to use for the majority vote.

### Random Forest
Random Forest classifier is an ensemble learning algorithm that uses a method called bagging to train a multitude of independent decision tree models to make a prediction using majority vote. Each decision tree takes samples from the original data. These samples can be chosen multiple times across multiple decision trees (this is known as “with replacement”). The graph below from Analytics Vidhya (Sruthi, 2023) provides a good visual representation of how the random forest algorithm works:

![image](https://user-images.githubusercontent.com/75294739/231309608-58208271-3a16-4d64-8d88-d8907ce90b46.png)

### Majority Vote
The Majority Vote algorithm is an ensemble learning technique that employs multiple independent machine learning models to predict classifications. In this experiment, a soft voting method will be used as the voting criteria. Unlike the hard voting method, which relies on the highest number of votes, the soft voting method selects the prediction based on the standalone algorithm with the highest overall probability. To conduct this experiment, I will select the top three performing standalone algorithms and keep the hyper-parameters that yield the best performance results.

### Gradient Boosting
The Gradient Boosting algorithm is another ensemble learning technique that combines multiple weak models iteratively to create a strong model that corrects errors of previous models. By itself, weak learning models are limited in their ability to make accurate predictions, but their iterative combination can lead to a highly accurate model prediction. Unlike the bagging technique used in random forest, this algorithm uses boosting technique, which improves predictive performance by building upon the strengths and weaknesses of previous models. Similar to logistic regression, gradient boosting uses log-loss function improve performance. Gradient boosting can also use binomial deviance loss:

Deviance = log(1+e^(-y_i f(x_i ) ) )

And Exponential Loss:

Exponential Loss = e^(-y_i f(x_i ) )

	Like the decision trees algorithm, gradient boosting can be susceptible to overfitting, especially if the number of estimators in the model are high (Corporate Finance Institute, 2023).
  
### Deep Learning Artificial Neural Network (ANN)
The artificial neural network (ANN) algorithm consists of interconnected layers and nodes (also known as perceptron) that stem from an initial input level and use forward and backward propagation techniques to increase model performance. These models can have multiple layers of nodes and can have significant numbers of neural connections. During the training stage, the ANN algorithm will set the initial weights and bias for each node, predict the output (forward propagation), compare the output to the actual prediction using a loss function, then re-adjust the weights by a specified learning rate (backwards propagation). These steps are completed repeatedly a specified number of times, known epochs, until the loss function has been minimized (Seth, 2021).  As with Logistic regression, the ANN algorithm uses sigmoid activation. Other activation functions include the ReLU (rectified linear unit) function:

ReLU=max(0,x)

And the Tanh function:

Tanh=  (e^x-e^(-x))/(e^x-e^(-x) )

## Performance Metrics Used
Since the class predictor variable in the Pima Indians diabetes dataset is a binary and is not equally distributed, the best performance metric to use would be the Area Under the Curve score (AUC). The AUC score takes the output of the confusion matrix of a learning algorithm and plots the true positive rate against the false positive rate. The AUC performance metric works has been shown to perform better at distinguishing between binary classes than accuracy score.

![image](https://user-images.githubusercontent.com/75294739/231309647-83a0c9fb-0efa-424c-aa3d-77fb17008396.png)

# Results of the Experiment
### Effects of Imputation on Feature Distribution
 
![image](https://user-images.githubusercontent.com/75294739/231309668-a6bad94d-bc7c-4e5e-9aa1-ca38b7a5ba40.png)
![image](https://user-images.githubusercontent.com/75294739/231309685-288a1efb-88e4-4ae3-979d-ba1cee84ff1b.png)
![image](https://user-images.githubusercontent.com/75294739/231309708-47ca2ab7-cea1-4d29-b954-8a1f404895ad.png)
![image](https://user-images.githubusercontent.com/75294739/231309720-4e6b2290-59b6-498b-a67d-3411e26eb92a.png)
 
In the original data, shown above, features such as Insulin and Skin Thickness have a pronounced right skewness due to a considerable proportion of the null data points being zeroed out. These features become more normalized across all the imputation methods, which can be seen in the charts below the original data distribution. Another unique distribution in the original distribution is on the Age feature. According to the study, there was an artificial age floor placed in the sample collection that created a minimum age of 21. The distribution of this feature skews right, with a significant proportion of patients sampled being around 21-30 years of age. Because the Age feature did not have any null values, the distribution of this variable stays the same across all the imputation methods. 

Some other observations that were identified across the three imputation distributions was how each model handled zero values in the Pregnancies feature. It appears that the linear regression imputation method kept a certain percentage of the data points as zero, while KNN and Random Forest kept none and distributed all of the zero values across the distribution. As stated above in the methodology section, due to the null values being zeroed out on the original dataset, it is impossible to determine which patients did have zero pregnancies and which ones had missing data. 

Another observation on the linear regression imputation distribution is in the Insulin feature. The left tail of the distribution curve dips into negative values, which is impossible since blood sugar cannot dip below 0. 

## Imputation Improvement by Learning Model

![image](https://user-images.githubusercontent.com/75294739/231309778-daf92709-6ebb-460d-b4ec-9d285429b84d.png)
![image](https://user-images.githubusercontent.com/75294739/231309072-7e784815-3d1f-4810-8062-5960d42a6ceb.png)

The graph above compares the performance of each learning model using the original data and the imputed data. Upon initial inspection, it is evident that the decision tree model had the poorest performance across all imputation methods. As discussed earlier in the methodology section, this could be attributed to the classifier imbalance, which may lead to issues with node splitting.

Upon closer examination of both the graph and the table, it becomes clear that most of the imputation methods resulted in improved performance compared to the original data, with a few exceptions. The decision tree model showed a decrease in performance of over 6.5% with both Linear Regression and Random Forest imputation methods, making the respective models only marginally better than randomly guessing the classifier.

## Best Cumulative Imputation Method

![image](https://user-images.githubusercontent.com/75294739/231309811-97c3289d-9dac-417e-baec-f238e0bae7bd.png)
![image](https://user-images.githubusercontent.com/75294739/231309236-cb44de40-7177-4204-892e-e95ae8dbfbd1.png)


One way to determine the best imputation method is to evaluate the improvement percent difference for each learning model using the best imputation method score. Based on the table above, the KNN model using linear regression imputation showed the highest improvement, followed by the ANN learning model using random forest imputation.

The chart above displays the cumulative improvement differential for each imputation method. Linear Regression was the top-performing method, with a cumulative score of 5.74% improvement. Random Forest imputation followed closely behind with a cumulative differential improvement of 4.11%, while KNN imputation had a differential of 2.02%. 
## Learning Model with Best Average Performance Improvement

![image](https://user-images.githubusercontent.com/75294739/231309351-91d49389-0a2e-4747-8065-ce4e32d6e537.png)

The table above displays the average improvement percentage across all learning models used on the dataset. The KNN model showed consistent improvement across all three different imputation methods, with an average improvement of 3.62%. In contrast, the decision tree learning model had an average percent improvement of -3.74%, and both linear and random forest imputation methods performed over 6.5% worse than the original data performance.

One plausible reason why linear regression imputation may not improve the performance of a decision tree model is that it assumes a linear relationship between feature variables when filling in null values. However, this assumption may not hold true in many cases, particularly when there are nonlinear relationships between variables. Additionally, the classifier imbalance issue mentioned earlier in this paper can lead to bias that favors the dominant classifier for both linear regression and random forest imputation methods, further impacting the performance of the decision tree model.
Some other learning models that showed negative average improvement across all imputation methods are random forest and gradient boosting. 

## Best Overall Learning Model Performance
The learning model that had the best overall performance across all imputation methods was the Keras artificial neural network, with an AUC score of 85.03% using the random forest imputation method. The hyper-parameters that were used in cross-validation to fine-tune the model are as follows:

![image](https://user-images.githubusercontent.com/75294739/231309459-87029fca-be1f-4603-8105-8ef3a9177485.png)

After cross validation was complete using the conditions above, the best combination of hyper-parameters that yielded the highest performance results on the training data was:

'activation': 'sigmoid', 'batch_size': 64, 'epochs': 150, 'hidden_layers': 3, 'neurons': 32, 'optimizer': 'adam'

Applied to the test data, the following fine-tuned hyper-parameters yielded the resulting ROC curve with accompanying AUC score as well as feature importance graph:

![image](https://user-images.githubusercontent.com/75294739/231309856-c035b048-4de3-42a7-834b-8a3ea174e1eb.png)
![image](https://user-images.githubusercontent.com/75294739/231309875-b33757fb-ac1a-4342-87be-c28ed3ee601e.png)

The graph above depicting feature importance shows that the glucose feature is the most crucial variable for predicting the diagnosis using the artificial neural network model. Interestingly, the feature importance ranking displayed in the graph aligns with the ranking obtained through the correlation matrix analysis conducted on the outcome variable, as detailed in the methodology section. 

![image](https://user-images.githubusercontent.com/75294739/231309917-11564e64-a9b3-4947-939d-b4aa8173a020.png)
 
## Best Hyper-Parameters by Learning Model and Imputation Method

![image](https://user-images.githubusercontent.com/75294739/231309940-a844e1d5-ca8b-4599-98ff-9bb389ab3914.png)
 
# Conclusion
In conclusion, this study aimed to evaluate the effects of imputation methods on feature distribution and the performance of learning models in diabetes prediction. The results indicate that most imputation methods resulted in improved performance compared to the original data, except for the decision tree model and, to a lesser extent, the gradient boosting model. The KNN model consistently showed improvement across all three imputation methods, and had the highest improvement score of all the learning models evaluated with a 4.05% improvement using linear regression imputation. The Keras artificial neural network with random forest imputation yielded the highest AUC score of 85.03%, with the glucose feature identified as the most important variable for predicting the diagnosis.

The study also identified unique feature observations in the original data distribution, such as the right skewness in the Insulin and Skin Thickness features due to a significant proportion of the null data points being zeroed out. The Age feature was found to have a right skewed distribution, with a significant proportion of patients being around 21-30 years of age. The study also highlighted how different imputation methods handled zero values in the Pregnancies feature and observed the left tail dipping into negative values in the Insulin feature for the linear regression imputation distribution.

Overall, the study provides insight into the effects of imputation methods on feature distribution and the performance of learning models. The results suggest that the KNN model with linear regression imputation is the best cumulative imputation method, while the Keras artificial neural network with random forest imputation is the best overall learning model for diabetes prediction. However, further research may be needed to evaluate the performance of different learning models and imputation methods on larger datasets and different medical conditions.


# References
- Bennett, P. H., Burch, T. A., & Miller, M. (1971). Diabetes Mellitus in American (Pima) Indians. The Lancet, 125-128.
- Bento, C. (2021, June 28). Decision Tree Classifier explained in real-life: picking a vacation destination. Retrieved from Towards Data Science: https://towardsdatascience.com/decision-tree-classifier-explained-in-real-life-picking-a-vacation-destination-6226b2b60575
- Biessels, G. J., & Despa, F. (2018). Cognitive decline and dementia in diabetes mellitus: mechanisms and clinical implications. Nature Reviews Endocrinology, Volume 14, 591-604.
- Biju, S. M., & James, K. C. (2022). Some Studies and Inferences on Pima Indian Diabetes Data. Journal of Statistics and Mathematical Engineering, 1-9.
- Brownlee, J. (2020, August 21). Cost-Sensitive Decision Trees for Imbalanced Classification. Retrieved from Machine Learning Mastery: https://machinelearningmastery.com/cost-sensitive-decision-trees-for-imbalanced-classification/#:~:text=The%20decision%20tree%20algorithm%20is,two%20groups%20with%20minimum%20mixing.
- Chang, V., Bailey, J., Xu, Q. A., & Sun, Z. (2022, March 24). Pima Indians diabetes mellitus classification based on machine learning (ML) algorithms. Retrieved from Neural Computing and Applications: https://doi.org/10.1007/s00521-022-07049-z
- Corporate Finance Institute. (2023, January 8). Gradient Boosting: A method used in building predictive models. Retrieved from Corporate Finance Institute: https://corporatefinanceinstitute.com/resources/data-science/gradient-boosting/
- Dash, S. K. (2022, September 22). Handling Missing Values with Random Forest. Retrieved from Analytics Vidhya: https://www.analyticsvidhya.com/blog/2022/05/handling-missing-values-with-random-forest/
- Faisal, S. (2018, March 29). 2.3 Nearest Neighbors Method. In Nearest Neighbor Methods for the Imputation of Missing Values in Low and High-Dimensional Data (pp. 14-15). Gottingen: Cuvillier Verlag. Retrieved from https://search-ebscohost-com.proxy.lib.wayne.edu/login.aspx?direct=true&db=e000xna&AN=2130272&site=ehost-live&scope=site
- Gandhi, R. (2018, May 5). Naive Bayes Classifier. Retrieved from Towards Data Science: https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c
- Inter Tribal Council of Arizona. (2023, March 30). Salt River Pima-Maricopa Indian Community. Retrieved from Inter Tribal Council of Arizona: https://itcaonline.com/member-tribes/salt-river-pima-maricopa-indian-community/#:~:text=Consisting%20of%2052%2C600%20acres%2C%20the,Tempe%2C%20Fountain%20Hills%20and%20Mesa.
- Ozsahin, D. U., Mustapha, M. T., Mubarak, A. S., Ameen, Z. S., & Uzun, B. (2022). Impact of Outliers and Dimensionality Reduction on the Performance of Predictive Models for Medical Disease Diagnosis. International Conference on Artificial Intelligence in Everything, 79-86.
- scikit learn. (2023). Decision Trees. Retrieved from scikit learn: https://scikit-learn.org/stable/modules/tree.html
- Seth, N. (2021, June 8). How does Backward Propagation Work in Neural Networks? Retrieved from Analytics Vidhya: https://www.analyticsvidhya.com/blog/2021/06/how-does-backward-propagation-work-in-neural-networks/
- Sruthi, E. R. (2023, March 24). Understand Random Forest Algorithms With Examples (Updated 2023). Retrieved from Analytics Vidhya: https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/
- Swalin, A. (2018, January 30). How to Handle Missing Data. Retrieved from Towards Data Science: https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
- Tan, P.-N., Steinbach, M., Karpatne, A., & Kumar, V. (2022). 6.10.6 Random Forests. In Introduction to Data Mining: Second Edition (pp. 512-514). Uttar Pradesh: Pearson.
- Thorn, J. (2020, February 8). Logistic Regression Explained. Retrieved from Towards Data Science: https://towardsdatascience.com/logistic-regression-explained-9ee73cede081
- UCI Machine Learning. (2023, March 25). Pima Indians Diabetes Database. Retrieved from Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- World Health Organization. (2022, September 16). Diabetes. Retrieved from World Health Organization: https://www.who.int/news-room/fact-sheets/detail/diabetes

