#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <assert.h>
using namespace std;
struct Connection {
	double weight;
	double deltaWeight;
};
class Neuron;
typedef vector<Neuron> Layer;
class Neuron {
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	Neuron(vector<double> &outputWeights, unsigned myIndex);
	void setOutputVal(double val) { n_outputVal = val; }
	double getOutputVal(void) const { return n_outputVal; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	double sumDOW(const Layer &nextLayer) const;
	void updateInputWeights(Layer &prevLayer);
	void softMax(vector<double> &outputVals);
	void storeNeuron(vector<double> &weights);
	static void changeEta(double n_eta) {
		Neuron::eta = n_eta;
	}
private:
	static double transferFunction(double x, vector<double> &n);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / (double(RAND_MAX)*10.0); }
	static double eta; // 0.0 to 1.0
	static double alpha; //0.0 to n
	static int batchSize;
	double n_outputVal;
	int batchCount;
	vector<Connection> n_outputWeights;
	unsigned n_myIndex;
	double n_gradient;
	vector<double> storeGradient;
};
double Neuron::eta = 0.75;
double Neuron::alpha = 0.5;
int Neuron::batchSize = 10;
void Neuron::updateInputWeights(Layer &prevLayer) {
	for (unsigned n = 0; n < prevLayer.size(); n++) {
		Neuron &neuron = prevLayer[n];
		if (storeGradient.size() < n + 1) {
			storeGradient.push_back(0.0);
		}
		storeGradient[n] += neuron.getOutputVal()*n_gradient;
		batchCount++;
		if (batchCount == ((batchSize - 1)*prevLayer.size()) + n + 1) {
			assert(!isnan(storeGradient[n]));
			neuron.n_outputWeights[n_myIndex].weight += eta * storeGradient[n] / (batchSize);
			storeGradient[n] = 0.0;
		}
	}
	if (batchCount == batchSize * prevLayer.size()) {
		batchCount = 0;
	}
}
double Neuron::sumDOW(const Layer &nextLayer) const {
	double sum = 0.0;
	for (unsigned n = 0; n < nextLayer.size() - 1; n++) {
		sum += n_outputWeights[n].weight * nextLayer[n].n_gradient;
	}
	return sum;
}
void Neuron::calcHiddenGradients(const Layer &nextLayer) {
	double dow = sumDOW(nextLayer);
	n_gradient = dow;
}
void Neuron::calcOutputGradients(double targetVals) {
	//n_gradient = ((targetVals / n_outputVal) - ((1 - targetVals) / (1 - n_outputVal))) * Neuron::transferFunctionDerivative(n_outputVal); //sigmoid cross entropy cost func
	//n_gradient = (targetVals - n_outputVal) * Neuron::transferFunctionDerivative(n_outputVal); //quad cost func
	n_gradient = targetVals - n_outputVal; //cross entropy cost func for both sigmoid and tanh
}
double Neuron::transferFunction(double x, vector<double> &n) {
	//return tanh(x);
	//return 1.0 / (1.0 + exp(-x));
	assert(n.size() > 0);
	double sum = 0;
	for (int i = 0; i < n.size(); i++) {
		sum += exp(n[i]);
	}
	return exp(x) / sum;
}
double Neuron::transferFunctionDerivative(double x) {
	//return 1.0-(x*x); //aprox. for domain 
	return  x - (x*x);
}
void Neuron::softMax(vector<double> &outputVals) {
	n_outputVal = transferFunction(n_outputVal, outputVals);
	assert(!isnan(n_outputVal));
}
void Neuron::feedForward(const Layer &prevLayer) {
	double sum = 0.0;
	for (unsigned n = 0; n < prevLayer.size(); n++) {
		sum += prevLayer[n].getOutputVal() * prevLayer[n].n_outputWeights[n_myIndex].weight;
	}
	n_outputVal = sum; //activation function
}
Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
	for (unsigned c = 0; c < numOutputs; c++) {
		n_outputWeights.push_back(Connection());
		n_outputWeights.back().weight = randomWeight();
	}
	n_myIndex = myIndex;
	batchCount = 0;
}
Neuron::Neuron(vector<double> &outputWeights, unsigned myIndex) {
	n_myIndex = myIndex;
	batchCount = 0;
	for (unsigned c = 0; c < outputWeights.size(); c++) {
		n_outputWeights.push_back(Connection());
		n_outputWeights.back().weight = outputWeights[c];
	}
}
void Neuron::storeNeuron(vector<double> &weights) {
	weights.clear();
	for (int w = 0; w < n_outputWeights.size(); w++) {
		weights.push_back(n_outputWeights[w].weight);
	}
}