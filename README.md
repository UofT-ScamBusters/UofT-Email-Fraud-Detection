# 👻 UofT ScamBusters: UofT Email Fraud Detection

* **A machine learning project built by:** Helena Glowacki, Lucia Kim, Alessia Ruberto
* **Read our full report here:** [UofT ScamBusters: UofT Email Fraud Detection - Report](https://github.com/UofT-ScamBusters/UofT-Email-Fraud-Detection/blob/main/final_report.pdf)

## 📧 Our Motivation

Have you ever received an email that seemed too good to be true? Perhaps it offered you a free gift or even a well-paying job opportunity.

If you answered *Yes*, you're not alone. Phishing continues to be the most common form of cyber crime, with **3.4 billion** phishing emails sent daily. 
A 2021 study revealed that phishing campaigns, on average, achieved a **17.8%** click rate. More focused spear phishing campaigns saw a significantly higher click rate, averaging **53.2%**.

Here at the University of Toronto, we've seen a significant increase in fraudulent emails targeting students, concerning the security of our personal information. In 2021, a staggering **40,000 student email accounts** were targeted by these malicious attacks.
Interestingly, the findings from our survey show that **80.6%** of students believe they are more than likely to be able to differentiate between a UofT and non-UofT phishing email, indicating a noticeable difference in these attacks.
Understanding the unique aspects of UofT-specific phishing emails can be crucial for developing more robust and tailored detection systems. 
Our goal is to highlight the importance of context-specific analysis in cybersecurity and give broader insight into phishing detection.

## 🔍 Understanding the Data

For our project, we are using **2 datasets** which we will refer to as:
1. **Non-UofT (Kaggle) data**
2. **UofT data**

These datasets are organized into **3 columns**: index, email text, and email type (phishing or safe) 

The word clouds below showcase the key differences between the two datasets. From this, we can observe that UofT has a specific “brand” of spam emails that targets students. 
More specifically, UofT phishing emails are typically related to scam job postings or UofT services, whereas the non-UofT (Kaggle) dataset includes a wide variety of general phishing emails.

<img src="https://github.com/UofT-ScamBusters/UofT-Email-Fraud-Detection/blob/main/visualizations/kaggle_phishing_wordcloud.png?raw=true" width="49%"/> <img src="https://github.com/UofT-ScamBusters/UofT-Email-Fraud-Detection/blob/main/visualizations/uoft_phishing_wordcloud.png?raw=true" width="49%"/> 

## 📑 Our Project Architecture

To preprocessing the datasets' email content, we removed URLs, special characters, normalized whitespace, and converted everything to lowercase. 
We then vectorized the data and performed dimensionality reduction, limiting the features to a maximum of 10,000 to prevent overfitting and mitigate the curse of dimensionality.
Finally, we split the data into 70% for training, 15% for testing, and 15% for validation. Below is our project architecture:

<img src="https://github.com/UofT-ScamBusters/UofT-Email-Fraud-Detection/blob/main/project_architecture.png?raw=true" /> 

## ✏️ Results

Our models were trained on the phishing email dataset found on Kaggle. We were interested in seeing how well the models would generalize when tested on UofT-specific emails without being trained on them. Below are the results for the accuracy of each model under each set of inputs:

| ML Algorithm              | Training Accuracy | Validation Accuracy | Test Accuracy (Non-UofT Data) | Test Accuracy (UofT Data) |
|---------------------------|-------------------|---------------------|------------------------------|--------------------------|
| Decision Tree             | 0.9282            | 0.8981              | 0.8848                       | 0.4500                   |
| Neural Network (SGD)      | 0.9549            | 0.9475              | 0.9498                       | 0.5833                   |
| Naive Bayes               | 0.9793            | 0.9733              | 0.9711                       | 0.6667                   |
| Logistic Regression       | 0.9866            | 0.9730              | 0.9738                       | 0.7167                   |
| Ensemble                  | 0.9838            | 0.9734              | 0.9692                       | 0.6667                   |



As we suspected, the accuracy of the models using the UofT test set is **significantly lower** than that of the non-UofT (Kaggle) test set! 🤔

Additionally, we noticed an interesting detail in the type of inaccuracies caused by the models. 
We will describe a “positive” result as an email flagged as phishing, “negative” otherwise. 
Analyzing the figures below, we notice a notably higher rate of false negatives than the false positive rate when the algorithms are tested on the UofT data.

<img src="https://github.com/UofT-ScamBusters/UofT-Email-Fraud-Detection/blob/main/visualizations/confusion_matrix_dt.png?raw=true" width="24%"/> <img src="https://github.com/UofT-ScamBusters/UofT-Email-Fraud-Detection/blob/main/visualizations/confusion_matrix_nn.png?raw=true" width="24%"/> <img src="https://github.com/UofT-ScamBusters/UofT-Email-Fraud-Detection/blob/main/visualizations/confusion_matrix_nb.png?raw=true" width="24%"/> <img src="https://github.com/UofT-ScamBusters/UofT-Email-Fraud-Detection/blob/main/visualizations/confusion_matrix_lr.png?raw=true" width="24%"/> 

The overwhelming majority of the UofT test set was classified as safe, regardless of the true label. **This raises a concern:** individuals who are less familiar with specific UofT phishing tactics like first-year students, are higher at risk. This may imply that general phishing detection models will not be very effective against the unique aspects of UofT-specific phishing emails.

## ✨ Takeaways

Overall, every model performed significantly better on the non-UofT (Kaggle) dataset compared to the UofT-specific dataset.
This further proves that **models trained on general phishing email data cannot be used to flag UofT-specific spam emails** due to their unique characteristics.

In the future, it would be interesting to expand the UofT-specific dataset that we are able to train on extensively and potentially improve model accuracy.
A potential strategy we could use to improve the model accuracy is utilizing natural language processing (NLP), since we noticed specific common keywords and phrases that are found in phishing emails that we could possibly identify. Moreover, our research proves that institutions like UofT with specifically-styled phishing email scams may need tailored phishing awareness training and uniquely trained machine learning models in order to successfully combat this problem! ✨

