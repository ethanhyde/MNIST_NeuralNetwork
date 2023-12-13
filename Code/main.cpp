#include "neuralNetwork.h"
#include <fstream>
#include <iostream>
#include <vector>
void loadMnistData(const std::string& image_filename, const std::string& label_filename, std::vector<std::vector<double>>& images, std::vector<std::vector<double>>& labels);

// Function to reverse integer bytes
int reverseInt(int i) 
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// load MNIST images
void loadMnistImages(const std::string& filename, std::vector<std::vector<double>>& images) 
{
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) 
    {
        int cols = 0;
        int number = 0;
        int numImages = 0;
        int rows = 0;
        
        file.read((char*)&number, sizeof(number));

        //reverse the bytes
        number = reverseInt(number);

        if (number != 2051) 
        {
            std::cout << "Invalid MNIST image :(" << std::endl;
            return;
        }

        file.read((char*)&numImages, sizeof(numImages)), numImages = reverseInt(numImages);
        file.read((char*)&rows, sizeof(rows)), rows = reverseInt(rows);
        file.read((char*)&cols, sizeof(cols)), cols = reverseInt(cols);

        for (int i = 0; i < numImages; ++i) 
        {
            std::vector<double> image(rows * cols);
            for (int k = 0; k < rows; ++k) 
            {
                for (int j = 0; j < cols; ++j) 
                {
                    unsigned char pixels = 0;
                    file.read((char*) &pixels, sizeof(pixels));
                    image[k * cols + j] = (double)pixels / 255.0; // normalize the pixels values 
                }
            }
            images.push_back(image);
        }
    }
    file.close();
}

// Function to load MNIST labels
void loadMnistLabels(const std::string& filename, std::vector<std::vector<double>>& labels) 
{
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) 
    {
        int number = 0;
        int numLabels = 0;

        file.read((char*)&number, sizeof(number));

        number = reverseInt(number);

        if (number != 2049) 
        {
            std::cout << "Invalid MNIST label file!" << std::endl;
            return;
        }

        file.read((char*)&numLabels, sizeof(numLabels)), numLabels = reverseInt(numLabels);

        for (int i = 0; i < numLabels; ++i) 
        {
            unsigned char label = 0;

            file.read((char*)&label, sizeof(label));
            std::vector<double> one_hot(10, 0.0); // i think this works?
            one_hot[label] = 1.0; 
            labels.push_back(one_hot);
        }
    }
    file.close();
}

// Combined function to load both images and labels
void loadMnistData(const std::string& image_filename, const std::string& label_filename, std::vector<std::vector<double>>& images, std::vector<std::vector<double>>& labels) 
{
    loadMnistImages(image_filename, images);
    loadMnistLabels(label_filename, labels);
}

// Assuming loadMnistData is defined as before
void loadMnistData(const std::string& image_filename, const std::string& label_filename, std::vector<std::vector<double>>& images, std::vector<std::vector<double>>& labels);

int main() {
    // Define file names for MNIST dataset
    std::string train_images_file = "MNIST_Dataset/train-images.idx3-ubyte";
    std::string train_labels_file = "MNIST_Dataset/train-labels.idx1-ubyte";
    std::string test_images_file = "MNIST_Dataset/t10k-images.idx3-ubyte";
    std::string test_labels_file = "MNIST_Dataset/t10k-labels.idx1-ubyte";


    // Change these to see if we can improve the accuracy

    // Parameters for the neural network
    int inputSize = 784;  // MNIST images are 28x28 pixelss, flattened
    int outputSize = 10;  // 10 classes for the digits 0-9
    int hiddenSize = 100; // Size of the hidden layers
    int hiddenLayers = 1; // Number of hidden layers
    double learningRate = 0.001;
    int epochs = 10;

    // Load MNIST training and testing data
    std::vector<std::vector<double>> mnist_train_images, mnist_train_labels;
    std::vector<std::vector<double>> mnist_test_images, mnist_test_labels;
    loadMnistData(train_images_file, train_labels_file, mnist_train_images, mnist_train_labels);
    loadMnistData(test_images_file, test_labels_file, mnist_test_images, mnist_test_labels);

    // Initialize neural network
    NeuralNetwork nn(inputSize, hiddenSize, outputSize, learningRate, hiddenLayers);

    std::cout << "Training Images: " << mnist_train_images.size() << ", Training Labels: " << mnist_train_labels.size() << std::endl;
    std::cout << "Test Images: " << mnist_test_images.size() << ", Test Labels: " << mnist_test_labels.size() << std::endl;


    // Training loop 
    for (size_t i = 0; i < mnist_train_images.size(); ++i) {
        for (int epoch = 0; epoch < epochs; ++epoch) 
        {
            std::cout <<"Image " << i << " - Epoch " << epoch << std::endl;
            //try {
                nn.train(mnist_train_images[i], mnist_train_labels[i]); // Train on each image-label pair
                std::cout << "Epoch " << (epoch + 1) << " completed." << std::endl;
            //}
            // catch (...) {
            //     std::cout << epoch << ": failed" << std::endl;
            // }
        
        }
    }

    // Evaluate the trained model on the test set
    int correctPredicts = 0;

    for (size_t i = 0; i < mnist_test_images.size(); ++i) 
    {
        std::vector<double> predicted = nn.predict(mnist_test_images[i]);
        int predLabel = std::distance(predicted.begin(), std::max_element(predicted.begin(), predicted.end()));
        int actualLabel = std::distance(mnist_test_labels[i].begin(), std::max_element(mnist_test_labels[i].begin(), mnist_test_labels[i].end()));

        if (predLabel == actualLabel) 
        {
            ++correctPredicts;
        }
    }

    // Before calculating accuracy
std::cout << "Total test images: " << mnist_test_images.size() << std::endl;
std::cout << "Correct predictions: " << correctPredicts << std::endl;

// Probably not necessary check but it's here

// Check for division by zero
if (!mnist_test_images.empty()) 
{
    double accuracy = static_cast<double>(correctPredicts) / mnist_test_images.size();
    std::cout << "Accuracy: " << accuracy * 100.0 << "%" << std::endl;
} 

else 
{
    std::cout << "Test dataset is empty." << std::endl;
}

    return 0;
}
