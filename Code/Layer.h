#ifndef UOREGON_CIS431531_F23_GROUP_LAYER_H
#define UOREGON_CIS431531_F23_GROUP_LAYER_H

#define IMAGE_DIM 28

#include <vector>
#include <random>

class Layer {
public:
    Layer(unsigned int inputs, unsigned int outputs);
    unsigned int inputs, outputs;
    float bias;

    std::vector<std::vector<double>> weight_matrix;
    std::vector<double> forward_propagation_serial(const std::vector<double>& input_vector);
    std::vector<double> forward_propagation_parallelized(const std::vector<double>& input_vector);
    std::vector<double> forward_propagation_parallelized_reduction(const std::vector<double>& input_vector);
    std::vector<double> forward_propagation_mkl(const std::vector<double>& input_vector);

    std::vector<double> back_propagation_serial(const std::vector<double>& input, const std::vector<double>& error, double learningRate);

private:
    double sigmoid(double x);
    double sigmoidDeriv(double x);
};


#endif //UOREGON_CIS431531_F23_GROUP_LAYER_H
