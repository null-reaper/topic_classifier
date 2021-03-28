# -*- coding: utf-8 -*-
"""
@author: Clive Gomes <cliveg@andrew.cmu.edu>
@description: Naive Bayes Classifier
"""

import sys
import numpy as np

class NaiveBayes(object):

    def __init__(self, trainingData):
        """
        Initializes a NaiveBayes object and the naive bayes classifier.
        :param trainingData: full text from training file as a single large string
        """
        
        # Process Input
        train_data = [document.split('\t') for document in trainingData.split('\n')]
        
        # Define Probability Dicts
        self.p_labels = {'RED': 0, 'BLUE': 0}
        self.p_word_given_label = {'RED': {}, 'BLUE':{}}
        
        # Count Words
        for document in train_data:
            label = document[0]
            tokens = document[1].split()
            
            self.p_labels[label] += 1
            
            for token in tokens:
                #Initialize Token Counts
                if token not in self.p_word_given_label['RED']:
                    self.p_word_given_label['RED'][token] = 0
                if token not in self.p_word_given_label['BLUE']:
                    self.p_word_given_label['BLUE'][token] = 0
                    
                # Increment Count
                self.p_word_given_label[label][token] += 1
        
        # Compute P(class) with add-one smoothing
        n_docs = sum(self.p_labels.values())
        n_labels = len(self.p_labels)
        for label in self.p_labels:
            self.p_labels[label] = np.log( (self.p_labels[label] + 1) / (n_docs + n_labels) )
            
        # Compute P(word | class) with add-one smoothing 
        n_words_given_label = {}
        n_unique_words_given_label = {}
        for label in self.p_word_given_label:
            n_words_given_label[label] = sum(self.p_word_given_label[label].values())
            n_unique_words_given_label[label] = len(self.p_word_given_label[label])
            
        for label in self.p_word_given_label:
            for word in self.p_word_given_label[label]:
                self.p_word_given_label[label][word] = np.log( (self.p_word_given_label[label][word] + 1) / (n_words_given_label[label] + n_unique_words_given_label[label]) ) 
                

    def estimateLogProbability(self, document):
        """
        Calculating the probability p(label|words) for the document 
        :param sentence: the test sentence, as a single string without label
        :return: a dictionary containing log probability for each category
        """
        probabilities = {'red': self.p_labels['RED'], 'blue': self.p_labels['BLUE']}
        
        for word in document.split():
            probabilities['red'] += self.p_word_given_label['RED'][word]   
            probabilities['blue'] += self.p_word_given_label['BLUE'][word]
            
        return probabilities

    def testModel(self, testData):
        """
        Evaluaing the model on test data
        :param testData: the test file as a single string
        :return: a dictionary containing each item as identified by the key
        """
        
        # Process Input
        test_data = [document.split('\t') for document in testData.split('\n')]
        
        # Initialize Counts
        n_correct = 0
        n_total = 0
        true_pos = {'RED': 0, 'BLUE': 0}
        false_pos = {'RED': 0, 'BLUE': 0}
        false_neg = {'RED': 0, 'BLUE': 0}
        
        # Evaluate Model
        for document in test_data:
            true_label = document[0]
            
            # Predict Label
            pred = self.estimateLogProbability(document[1])
            pred_label = 'RED' if pred['red'] > pred['blue'] else 'BLUE'
            
            # Update Counts
            
            n_total += 1
            
            if pred_label == true_label:
                n_correct += 1
                
            for label in ['RED', 'BLUE']:
                if true_label == label and pred_label == label:
                    true_pos[label] += 1
                elif true_label != label and pred_label == label:
                    false_pos[label] += 1
                elif true_label == label and pred_label != label:
                    false_neg[label] += 1
                
        # Compute Metrics
        accuracy = n_correct / n_total
        
        prec_red = true_pos['RED'] / ( true_pos['RED'] + false_pos['RED'] )
        prec_blue = true_pos['BLUE'] / ( true_pos['BLUE'] + false_pos['BLUE'] )
        
        rec_red = true_pos['RED'] / ( true_pos['RED'] + false_neg['RED'] )
        rec_blue = true_pos['BLUE'] / ( true_pos['BLUE'] + false_neg['BLUE'] )

        return {'overall accuracy': accuracy,
                'precision for red': prec_red,
                'precision for blue': prec_blue,
                'recall for red': rec_red,
                'recall for blue': rec_blue}


# Main Routine
if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Usage: python3 naivebayes.py TRAIN_FILE_NAME TEST_FILE_NAME")
        sys.exit(1)

    train_txt = sys.argv[1]
    test_txt = sys.argv[2]

    with open(train_txt, 'r', encoding='utf8') as f:
        train_data = f.read()

    with open(test_txt, 'r', encoding='utf8') as f:
        test_data = f.read()

    model = NaiveBayes(train_data)
    evaluation = model.testModel(test_data)
    print("overall accuracy: " + str(evaluation['overall accuracy'])
        + "\nprecision for red: " + str(evaluation['precision for red'])
        + "\nprecision for blue: " + str(evaluation['precision for blue'])
        + "\nrecall for red: " + str(evaluation['recall for red'])
        + "\nrecall for blue: " + str(evaluation['recall for blue']))


