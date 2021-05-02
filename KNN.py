import pandas
import numpy
from sys import maxsize

class Point:
    def __init__(self, Class, distance = maxsize):
        self.Class = Class
        self.distance = distance

def getMaxOccurencePoint(Points, order):
    nOccurences = {}
    for i in range(len(Points)):
        if Points[i] not in nOccurences:
            nOccurences[Points[i]] = 1
        else:
            nOccurences[Points[i]] += 1
    maxOccurencePoint = None
    for Point in nOccurences:
        if maxOccurencePoint is None:
            maxOccurencePoint = Point
        else:
            if nOccurences[Point] > nOccurences[maxOccurencePoint]:
                maxOccurencePoint = Point
            elif nOccurences[Point] == nOccurences[maxOccurencePoint]:
                if order[Point.Class] < order[maxOccurencePoint.Class]:
                    maxOccurencePoint = Point
    return maxOccurencePoint
        
# Read and initialize training and testing data sets
trainSet = pandas.read_csv("TrainData.txt", header = None)
testSet = pandas.read_csv("TestData.txt", header = None)

# Split data sets into attributes and classes
trainAttributes = trainSet.drop(8, axis = 1)
trainClasses = trainSet[8]

testAttributes = testSet.drop(8, axis = 1)
testClasses = testSet[8]

print(len(testAttributes))

# Determine the order of each class in training set to break ties
order = {}
currentOrder = 0
for i in range(trainClasses.size):
    if trainClasses[i] not in order:
        order[trainClasses[i]] = currentOrder
        currentOrder += 1

# Compute Euclidean distance and accuracy of each k-value
for k in range(1, 10):
    nCorrect = 0
    print("k value: {}".format(k))
    for i in range(len(testAttributes)):
        minDistancePoints = [Point("")] * k
        for j in range(len(trainAttributes)):
            distance = numpy.linalg.norm(testAttributes.iloc[i]-trainAttributes.iloc[j])
            tempPoint = Point(trainClasses[j], distance)
            for tempk in range(k):
                if tempPoint.distance < minDistancePoints[tempk].distance:
                    tempPoint = minDistancePoints[tempk]
                    minDistancePoints[tempk] = Point(trainClasses[j], distance)
        predictedPoint = getMaxOccurencePoint(minDistancePoints, order)
        if predictedPoint.Class == testClasses[i]:
            nCorrect += 1
        print("Predicted class: {}, Actual class: {}".format(predictedPoint.Class, testClasses[i]))
    print("Number of correctly classified instances: {}, Total number of instances: {}\n".format(nCorrect, len(testClasses)))
    print("Accuracy: {}".format(float(nCorrect/len(testClasses))))
