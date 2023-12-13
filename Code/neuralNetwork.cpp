#include "neuralNetwork.h"
#include <iostream>

unsigned int MNIST_IMAGE_SIZE = IMAGE_DIM * IMAGE_DIM; // Definition

NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate, int numLayers)
{
    // add input layer to network
    network_layers.emplace_back(Layer(inputSize, hiddenSize));

    // add hidden layers
    for (int i = 0; i < numLayers; i++) {
        network_layers.emplace_back(hiddenSize, hiddenSize);
    }

    // add output layer
    network_layers.emplace_back(hiddenSize, outputSize);
}

NeuralNetwork::~NeuralNetwork() 
{
    // free any allocated memory
}

void NeuralNetwork::train(const vector<double>& input, const vector<double>& target)
 {
    // Forward pass
    std::vector<double> out = forward(input);
    std::cout << "Completed forward pass" << std::endl;
    
    // Compute the loss 
    double loss = computeLoss(out, target);
    std::cout << "Completed loss calculation" << std::endl;

    back_propagate(input, target);
    std::cout << "Completed back propagation" << std::endl;
}

std::vector<double> NeuralNetwork::forward(std::vector<double> input) 
{
    for (auto &layer: network_layers) 
    {
        input = layer.forward_propagation_serial(input); // Serial implementation only for now
    }
    return input; // The final output 
}


void NeuralNetwork::back_propagate(const vector<double>& input, const vector<double>& target) 
{

    // error calc
    vector<double> error = computeOutputLayerError(input, target);

    // backward pass
    for (int i = network_layers.size() - 1; i >= 0; --i) 
    {
        error = network_layers[i].back_propagation_serial(input, error, learningRate);
    }
}

vector<double> NeuralNetwork::computeOutputLayerError(const vector<double>& output, const vector<double>& target) 
{
    vector<double> error;
    for (int i = 0; i < output.size(); ++i) 
    {
        // Assuming mean squared error loss function
        double derivative = output[i] - target[i]; // Derivative of the loss function
        error.push_back(derivative);
    }

    return error;
}


double NeuralNetwork::computeLoss(const vector<double>& output, const vector<double>& target)
{
    double loss = 0.0;

    for (int i = 0; i < output.size(); ++i) 
    {
        loss += pow(target.at(i) - output.at(i), 2);
    }
    return loss / outputSize; // Returns the average loss
}


