# !wget --user=username --ask-password https://course.cse.ust.hk/comp2211/assignments/pa1/data/20ng_train_dataset.npz
# !wget --user=username --ask-password https://course.cse.ust.hk/comp2211/assignments/pa1/data/20ng_test_dataset.npz
# !wget --user=username --ask-password https://course.cse.ust.hk/comp2211/assignments/pa1/data/20ng_train_labels.npy
# !wget --user=username --ask-password https://course.cse.ust.hk/comp2211/assignments/pa1/data/20ng_test_labels.npy

import numpy as np

class NaiveBayesClassifier:
  def __init__(self, train_dataset, test_dataset, train_labels, test_labels):
    self.train_dataset = train_dataset
    self.test_dataset = test_dataset
    self.train_labels = train_labels
    self.test_labels = test_labels

  def build_training_delta_matrix(self):
    d = np.shape(self.train_labels)[0]   # number of training documents
    N = np.amax(self.train_labels) + 1   # number of classes

    deltas = (np.arange(N) == self.train_labels.reshape(-1, 1))

    return deltas

  def estimate_class_probabilities(self):
    deltas = self.build_training_delta_matrix()
    class_count = np.sum(deltas, axis=0) + 1
    class_prob = class_count / np.sum(class_count)
    return class_prob

  def estimate_word_probabilities(self):
    deltas = self.build_training_delta_matrix()
    word_count = np.dot(self.train_dataset.transpose(), deltas) + 1
    word_prob = word_count / np.sum(word_count, axis = 0)
    return word_prob

  def predict(self):
    class_prob = self.estimate_class_probabilities()
    word_prob = self.estimate_word_probabilities()
    log_class_prob = np.log(class_prob)
    log_word_prob = np.log(word_prob)
    log_posterio = log_class_prob + np.dot(self.test_dataset, log_word_prob)
    test_predict = np.argmax(log_posterio, axis=1)
    return test_predict


def generate_confusion_matrix(test_predict, test_labels):
  d_test = np.shape(test_labels)[0]
  N = np.amax(test_labels) + 1

  predict = np.arange(N) == test_predict.reshape(-1, 1)
  labels = np.arange(N) == test_labels.reshape(-1, 1)
  TP = ((labels == 1) & (predict == 1)).sum(axis=0)
  TN = ((labels == 0) & (predict == 0)).sum(axis=0)
  FP = ((labels == 0) & (predict == 1)).sum(axis=0)
  FN = ((labels == 1) & (predict == 0)).sum(axis=0)

  return TP, TN, FP, FN

def calculate_precision(test_predict, test_labels):
  TP, TN, FP, FN = generate_confusion_matrix(test_predict, test_labels)
  precision = np.sum(TP) / np.sum(TP + FP)
  return precision

def calculate_recall(test_predict, test_labels):
  TP, TN, FP, FN = generate_confusion_matrix(test_predict, test_labels)
  recall = np.sum(TP) / np.sum(TP + FN)
  return recall

def calculate_micro_f1(test_predict, test_labels):
  precision = calculate_precision(test_predict, test_labels)
  recall = calculate_recall(test_predict, test_labels)
  micro_f1 = 2 * precision * recall / (precision + recall)
  return micro_f1

def calculate_macro_f1(test_predict, test_labels):
  TP, TN, FP, FN = generate_confusion_matrix(test_predict, test_labels)
  p = TP / (TP + FP)
  r = TP / (TP + FN)
  macro_f1 = 2 * p * r / (p + r)
  macro_f1 = np.average(macro_f1)
  return macro_f1

"""# Optional Task: Test Run
Use all the functions we have previously defined in Tasks 1 and 2 to perform Naive Bayes text classifier on our 20 Newsgroups dataset. Feel free to modify this code cell for your own testing and debugging purposes, it will not be graded.
"""

import scipy.sparse as sparse

if __name__ == '__main__':
  train_dataset = sparse.load_npz("20ng_train_dataset.npz")
  test_dataset = sparse.load_npz("20ng_test_dataset.npz")
  train_dataset = train_dataset.toarray()
  test_dataset = test_dataset.toarray()
  train_labels = np.load("20ng_train_labels.npy")
  test_labels = np.load("20ng_test_labels.npy")

  Classifier = NaiveBayesClassifier(train_dataset, test_dataset, train_labels, test_labels)
  deltas = Classifier.build_training_delta_matrix()
  class_prob = Classifier.estimate_class_probabilities()
  word_prob = Classifier.estimate_word_probabilities()
  test_predict = Classifier.predict()

  TP, TN, FP, FN = generate_confusion_matrix(test_predict, test_labels)
  precision = calculate_precision(test_predict, test_labels)
  recall = calculate_recall(test_predict, test_labels)
  micro_f1 = calculate_micro_f1(test_predict, test_labels)
  macro_f1 = calculate_macro_f1(test_predict, test_labels)