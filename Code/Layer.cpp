#include "Layer.h"
#include <cmath>
#include <iostream>
// #include <mkl.h> // intel math library


Layer::Layer(unsigned int inputs, unsigned int outputs) : inputs(inputs), outputs(outputs) {
    // randomize weights and biases on initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0); // Random values between -1 and 1
    weight_matrix.resize(outputs, std::vector<double>(inputs));


    // weights
    for (int i = 0; i < weight_matrix.size(); ++i)
    {
        for (int j = 0; j < weight_matrix.at(i).size(); ++j)
        {
            weight_matrix.at(i).at(j) = dis(gen);
        }
    }

    // bias
    bias = (float)dis(gen);
}

double Layer::sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double Layer::sigmoidDeriv(double x)
{
    double sig = sigmoid(x);
    return sig * (1 - sig);
}

std::vector<double> Layer::forward_propagation_serial(const std::vector<double> &input_vector) {

    // std::cout << "Input vector size: " << input_vector.size() << std::endl;

    if (input_vector.empty()) 
    {
        std::cout << "Error: Input vector is empty." << std::endl;
        return std::vector<double>();
    
    }

    std::vector<double> output_vector(outputs, 0.0);

    for (int i = 0; i < outputs; i++) 
    {
        double val = 0;

        for (int j = 0; j < inputs; j++) 
        {
            val += input_vector.at(j) * weight_matrix.at(i).at(j);
        }
        output_vector[i] = sigmoid(val + bias);
    }
    return output_vector;
}


std::vector<double> Layer::forward_propagation_parallelized(const std::vector<double> &input_vector) {
    std::vector<double> output_vector;
    double value;

    #pragma omp parallel for schedule(static) shared(input_vector, output_vector)
    for (int i = 0; i < outputs; i++) {
        value = 0;
        for (int j = 0; j < inputs; j++) {
            value += input_vector.at(j) * weight_matrix.at(i).at(j);
        }
        output_vector.at(i) = sigmoid(value + bias);
    }
    return output_vector;
}

std::vector<double> Layer::forward_propagation_mkl(const std::vector<double> &input_vector) {

    std::vector<double> output_vector(input_vector.size());
    // do vector multiplication
    // cblas_dgemv(CblasRowMajor, CblasNoTrans, 
    //     inputs, outputs, 1.0, weight_matrix.data(), inputs, input_vector.data(), 1, 1.0, output_vector.data(), 1);
    return output_vector;
}



std::vector<double> Layer::back_propagation_serial(const std::vector<double>& input, const std::vector<double>& error, double learningRate) 
{

    std::vector<double> gradientWeight(outputs, 0.0);
    std::vector<double> gradientBias(outputs, 0.0);

    // Calculate gradients for weights and biases
    for (unsigned int i = 0; i < outputs; ++i) 
    {
        for (unsigned int j = 0; j < inputs; ++j) 
        {
            // Gradient for weight is product of error and input
            gradientWeight[i] += error[i] * input[j];
        }
        // Gradient for bias is simply the error
        gradientBias[i] += error[i];
    }

    // Update weights and biases
    for (unsigned int i = 0; i < outputs; ++i) 
    {
        for (unsigned int j = 0; j < inputs; ++j) 
        {
            weight_matrix[i][j] -= learningRate * gradientWeight[i];
        }
        bias -= learningRate * gradientBias[i];
    }

    // Calculate error for the previous layer
    std::vector<double> prevLayerError(inputs, 0.0);
    for (unsigned int i = 0; i < inputs; ++i) 
    {
        for (unsigned int j = 0; j < outputs; ++j) 
        {
            prevLayerError[i] += weight_matrix[j][i] * error[j];
        }
    }

    return prevLayerError;
}
