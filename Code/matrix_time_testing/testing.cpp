#include "../Code/Layer.h"
#include <iostream>
#include <chrono>

using namespace std::chrono;
 
int main() {

    // initialize layer with weight matrix to do a calculation on
    Layer test_layer = Layer(128, 128); // size of 128 (common hidden layer size)


    time_point<system_clock> start, end;
    duration<double> time_elapsed;
    start = high_resolution_clock::now();
    test_layer.forward_propagation_serial();
    end = high_resolution_clock::now();
 
    time_elapsed = end - start;

    std::cout << "Serial: " << time_elapsed.count() << std::endl;
    
    test_layer = Layer(128, 128);

    start = high_resolution_clock::now();
    test_layer.forward_propagation_parallelized();
    end = high_resolution_clock::now();
 
    time_elapsed = end - start;

    std::cout << "Parallelized: " << time_elapsed.count() << std::endl;

    test_layer = Layer(128, 128);

    start = high_resolution_clock::now();
    test_layer.forward_propagation_mkl();
    end = high_resolution_clock::now();
 
    time_elapsed = end - start;

    std::cout << "MKL Parallel: " << time_elapsed.count() << std::endl;
}