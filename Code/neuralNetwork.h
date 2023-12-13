#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <cmath>

#include "Layer.h"
using namespace std;

extern unsigned int MNIST_IMAGE_SIZE;

class NeuralNetwork
{
private:
    int inputSize = MNIST_IMAGE_SIZE;      // Size of the input layer (784 for standard MNIST image)
    int outputSize;     // Size of the output layer
    double learningRate; // Learning rate

    vector<Layer> network_layers;

public:
    // Constructor and destructor
    // NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate, int numLayers)
    //    : inputSize(inputSize), outputSize(outputSize), learningRate(learningRate) {};

    // I changed the constructor so it wasn't being defined twice
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate, int numLayers);
    ~NeuralNetwork();

    // Public methods
    void initWeights();
    vector<double> forward(const vector<double> input);

    // Training functions
    void train(const vector<double>& input, const vector<double>& target);
    double computeLoss(const vector<double>& output, const vector<double>& target);
    void back_propagate(const vector<double>& input, const vector<double>& target);
    void updateWeights();
    vector<double> computeOutputLayerError(const vector<double>& output, const vector<double>& target);
    vector<double> predict(const vector<double>& input) 
    {
        return forward(input);
    }
};

#endif
