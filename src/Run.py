#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from data.data_set import DataSet
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=False)
    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    
    myPerceptronClassifier = Perceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=30)
                                        
    myLRClassifier = LogisticRegression(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=30)

    myMLPClassifier = MultilayerPerceptron(data.trainingSet,
                                           data.validationSet,
                                           data.testSet,
                                           loss = 'bce',
                                           learningRate=0.1,
                                           epochs=25)
                                        
    
    # Report the result #
    print("=========================")
    evaluator = Evaluator()                                        

    # Train the classifiers
    print("=========================")
    print("Training..")


    print("\nMulti Layer Perceptron has been training..")
    myMLPClassifier.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    mlpPred = myMLPClassifier.evaluate()
    
    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("\nResult of the Multi Layer Perceptron recognizer:")
    evaluator.printAccuracy(data.testSet, mlpPred)
    
    # Draw
    plot = PerformancePlot("Mulit Layer Perceptron validation")
    plot.draw_performance_epoch(myMLPClassifier.performances,
                                myMLPClassifier.epochs)

    
    
if __name__ == '__main__':
    main()
