# ## Our Mission ##
# 
# Spam detection is one of the major applications of Machine Learning in the interwebs today. Pretty much all of the major email service providers have spam detection systems built in and automatically classify such mail as 'Junk Mail'. 
# 
# In this mission we will be using the Naive Bayes algorithm to create a model that can classify SMS messages as spam or not spam, based on the training we give to the model. It is important to have some level of intuition as to what a spammy text message might look like. Often they have words like 'free', 'win', 'winner', 'cash', 'prize' and the like in them as these texts are designed to catch your eye and in some sense tempt you to open them. Also, spam messages tend to have words written in all capitals and also tend to use a lot of exclamation marks. To the human recipient, it is usually pretty straightforward to identify a spam text and our objective here is to train a model to do that for us!
# 
# Being able to identify spam messages is a binary classification problem as messages are classified as either 'Spam' or 'Not Spam' and nothing else. Also, this is a supervised learning problem, as we will be feeding a labelled dataset into the model, that it can learn from, to make future predictions. 
# 
# # Overview
# 
# This project has been broken down in to the following steps: 
# 
# - Step 0: Introduction to the Naive Bayes Theorem
# - Step 1.1: Understanding our dataset
# - Step 1.2: Data Preprocessing
# - Step 2.1: Bag of Words (BoW)
# - Step 2.2: Implementing BoW from scratch
# - Step 2.3: Implementing Bag of Words in scikit-learn
# - Step 3.1: Training and testing sets
# - Step 3.2: Applying Bag of Words processing to our dataset.
# - Step 4.1: Bayes Theorem implementation from scratch
# - Step 4.2: Naive Bayes implementation from scratch
# - Step 5: Naive Bayes implementation using scikit-learn
# - Step 6: Evaluating our model
# - Step 7: Conclusion





import pandas as pd
# Dataset available using filepath 'smsspamcollection/SMSSpamCollection'
df = pd.read_table('smsspamcollection/SMSSpamCollection', names = ['label', 'sms_message'])

# Output printing out first 5 rows
df.head()


# ### Step 1.2: Data Preprocessing ###



df['label'] = df.label.map({'ham':0, 'spam':1})


# ### Step 2.1: Bag of Words ###

documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []
for i in documents:
    lower_case_documents.append(str.lower(i))
print(lower_case_documents)




sans_punctuation_documents = []
import string
punctuation = string.punctuation
for i in lower_case_documents:
    for element in i:
        if element in punctuation:
            i = i.replace(element, "")
    sans_punctuation_documents.append(i)
    
print(sans_punctuation_documents)


# **Step 3: Tokenization**
# 
# Tokenizing a sentence in a document set means splitting up the sentence into individual words using a delimiter. The delimiter specifies what character we will use to identify the beginning and  end of a word. Most commonly, we use a single space as the delimiter character for identifying words, and this is true in our documents in this case also.


preprocessed_documents = []
for i in sans_punctuation_documents:
    preprocessed_documents.append(str.split(i))
print(preprocessed_documents)


# **Step 4: Count frequencies**
# 
# Now that we have our document set in the required format, we can proceed to counting the occurrence of each word in each document of the document set. We will use the `Counter` method from the Python `collections` library for this purpose. 

frequency_list = []
import pprint
from collections import Counter

for i in preprocessed_documents:
    frequency_list.append(Counter(i))
    
pprint.pprint(frequency_list)


## Now with Sklearn
documents = ['Hello, how are you!',
                'Win money, win from home.',
                'Call me now.',
                'Hello, Call hello you tomorrow?']



from sklearn.feature_extraction.text import CountVectorizer

#Create Class instance for vectorizer
count_vector = CountVectorizer()


# **Data preprocessing with CountVectorizer()**


print(count_vector)


count_vector.fit(documents)
count_vector.get_feature_names()

doc_array = count_vector.transform(documents).toarray()
doc_array



frequency_matrix = pd.DataFrame(doc_array, columns = count_vector.get_feature_names())
frequency_matrix


# split into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))


# ### Step 3.2: Applying Bag of Words processing to our dataset. ###

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

# P(D)
p_diabetes = 0.01

# P(~D)
p_no_diabetes = 0.99

# Sensitivity or P(Pos|D)
p_pos_diabetes = 0.9

# Specificity or P(Neg|~D)
p_neg_no_diabetes = 0.9

# P(Pos)
p_pos = (p_diabetes * p_pos_diabetes) + (p_no_diabetes * (1-p_neg_no_diabetes))
print('The probability of getting a positive test result P(Pos) is: {}',format(p_pos))

# P(D|Pos)
p_diabetes_pos = (p_diabetes * p_pos_diabetes) / p_pos
print('Probability of an individual having diabetes, given that that individual got a positive test result is:',format(p_diabetes_pos)) 

# P(Pos|~D)
p_pos_no_diabetes = 0.1

# P(~D|Pos)
p_no_diabetes_pos = (p_no_diabetes * (1-p_neg_no_diabetes))/ p_pos
print('Probability of an individual not having diabetes, given that that individual got a positive test result is:',p_no_diabetes_pos)


 Step 1

# P(J)
p_j = 0.5

# P(F/J)
p_j_f = 0.1

# P(I/J)
p_j_i = 0.1

p_j_text = p_j_f * p_j_i
print(p_j_text)

# P(G)
p_g = 0.5

# P(F/G)
p_g_f = 0.7

# P(I/G)
p_g_i = 0.2

p_g_text = p_g_f * p_g_i
print(p_g_text)


p_f_i = p_g_text + p_j_text
print('Probability of words freedom and immigration being said are: ', format(p_f_i))

p_j_fi = (p_j * p_j_f * p_j_i)/p_f_i
print('The probability of Jill Stein saying the words Freedom and Immigration: ', format(p_j_fi))


p_g_fi = (p_g * p_g_f * p_g_i)/ p_f_i
print('The probability of Gary Johnson saying the words Freedom and Immigration: ', format(p_g_fi))



# ### Step 5: Naive Bayes implementation using scikit-learn ###


from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

predictions = naive_bayes.predict(testing_data)



# ### Step 6: Evaluating our model ###

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(predictions, y_test)))
print('Precision score: ', format(precision_score(predictions, y_test)))
print('Recall score: ', format(recall_score(predictions, y_test)))
print('F1 score: ', format(f1_score(predictions, y_test)))


# ### Step 7: Conclusion ###
# 
# One of the major advantages that Naive Bayes has over other classification algorithms is its ability to handle an extremely large number of features. In our case, each word is treated as a feature and there are thousands of different words. Also, it performs well even with the presence of irrelevant features and is relatively unaffected by them. The other major advantage it has is its relative simplicity. Naive Bayes' works well right out of the box and tuning its parameters is rarely ever necessary, except usually in cases where the distribution of the data is known. 
# It rarely ever overfits the data. Another important advantage is that its model training and prediction times are very fast for the amount of data it can handle. All in all, Naive Bayes' really is a gem of an algorithm!





