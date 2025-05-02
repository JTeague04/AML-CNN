# To train the model:

# For every email,
#   for each word,
#       preprocess the word
#       add each it to the relevant frequency dictionary
#       and add 1 to the counter for the number of spam/not spam words.


# To check an email for spam:

# Look at each word,
#   divide its frequency in spam and not spam, by the counter for both
#   (so if it consists of 1% of spam an 99% of not spam, then its not spam.)

import pandas as pd
data = pd.read_csv('spam_detection_training_data.csv')

emails = data['text']
labels = data['label']

spam = {}
spamcounter = 1
notspam = {}
notspamcounter = 1

TRAIN_AMOUNT = 2700

# Preprocessing function done per word
def preprocess(word):
    # If it's numeric, mark as such
    if word.isnumeric():
        return "NUM"
    return word

# TRAINING ===========================================================

def learn_from(email, is_spam):
    global spamcounter, notspamcounter
    # Check every word,
    for word in email.split():
        # Preprocess the word to ensure it has a key in the dictionary
        word = preprocess(word)
        if word == None: continue

        # Add the word to its relevant frequency dictionary
        if not word in spam: spam[word] = 0
        if not word in notspam: notspam[word] = 0

        if is_spam:
            spam[word] += 1
            spamcounter += 1
        else:
            notspam[word] += 1
            notspamcounter += 1

# For every email,
for index in range(len(emails[:TRAIN_AMOUNT])):
    # Learn from it
    learn_from(emails[index], labels[index])
            
# TESTING ===========================================================

def is_spam(email):
    spam_count = 0
    # Look at each word,
    for word in email.split():
        # Ensure the word is in the dictionary
        word = preprocess(word)
        if word == None: continue
        if not (word in spam or word in notspam): continue

        # Divide its frequency in spam and not spam, by the counter for both
        spam_proportion = spam[word] /spamcounter
        notspam_proportion = notspam[word] /notspamcounter
    
        # (so if it consists of 1% of spam an 99% of not spam, then its not spam.)
        spam_count += 1 if spam_proportion > notspam_proportion else -1

    return spam_count > 0

correct, incorrect = 0, 0

# Testing set
for index in range(len(emails[TRAIN_AMOUNT:])):
    
    if labels[index] == is_spam(emails[index]):
        correct += 1
    else:
        incorrect += 1

input(f"Accuracy: {correct *100 /(correct +incorrect)}%")




        
