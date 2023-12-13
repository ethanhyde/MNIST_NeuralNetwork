neural_network:
	g++ -std=c++11 main.cpp neuralNetwork.cpp Layer.cpp -o neural_network

parallel:
	g++ -std=c++11 -fopenmp -lpthread main.cpp neuralNetwork.cpp Layer.cpp -o neural_network 

clean:
	rm neural_network