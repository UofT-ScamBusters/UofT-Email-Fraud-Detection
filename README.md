# UofT ScamBusters: UofT Email Fraud Detection

**A machine learning project built by:** Helena Glowacki, Lucia Kim, Alessia Ruberto

**Read our full report here:** [UofT ScamBusters: UofT Email Fraud Detection - Report](FILE_NAME.pdf)

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

<img src="https://github.com/UofT-ScamBusters/UofT-Email-Fraud-Detection/blob/main/visualizations/kaggle_phishing_wordcloud.png?raw=true" width="425"/> <img src="https://github.com/UofT-ScamBusters/UofT-Email-Fraud-Detection/blob/main/visualizations/uoft_phishing_wordcloud.png?raw=true" width="425"/> 

## 📑 Our Plan

After preprocessing the datasets by removing URLs, special characters, normalizing whitespace, and conve OMG MY WORKLAPTOP IS UPDATING LET ME COMMIT

Removing URLs: Any URLs present in the email content were removed to prevent their influence on the model. 
Removing Special Characters: All special characters were stripped from the text, leaving only letters and numbers. 
Normalizing Whitespace: Multiple, trailing, and leading spaces were removed with a single space to clean up the text and make it consistent.
Converting to Lowercase: Finally, all the text was converted to lowercase, to ensure that the model treats words like "Email" and "email" as the same word.

After preprocessing, the cleaned email content was vectorized, converting the words into numerical features that the models could use to identify patterns. To prevent overfitting and mitigate the curse of dimensionality, we also performed dimensionality reduction, limiting the features to a maximum of 10,000. Finally, we split the data into 70% for training, 15% for testing, and 15% for validation.
