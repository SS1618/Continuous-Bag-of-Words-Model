#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <assert.h>
#include "Neuron.h"
using namespace std;
class Net {
public:
	Net(const unsigned inputSize, const unsigned outputSec, const unsigned windowSize);
	Net(const string &fileName);
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;
	void getResults(vector<double> &resultVals, unsigned layer) const;
	void storeNet(const string &fileName);

private:
	vector<Layer> m_layers; //m_layer[layerNumber][neuronInLayer]
	double n_error;
	double n_recentAverageError;
	static double n_recentAverageSmoothingFactor;
	unsigned outputSections;
};
double Net::n_recentAverageSmoothingFactor = 18000.0;
void Net::getResults(vector<double> &resultVals) const {
	resultVals.clear();
	for (unsigned n = 0; n < m_layers.back().size() - 1; n++) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}
void Net::getResults(vector<double> &resultVals, unsigned layer) const {
	assert(layer >= 0 && layer < m_layers.size());
	resultVals.clear();
	for (unsigned n = 0; n < m_layers[layer].size() - 1; n++) {
		resultVals.push_back(m_layers[layer][n].getOutputVal());
	}
}
void Net::backProp(const vector<double> &targetVals) {
	//calculated overall net error
	Layer &outputLayer = m_layers.back();
	//calc output gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
		Layer &prevLayer = m_layers[m_layers.size() - 2];
		outputLayer[n].calcOutputGradients(targetVals[n]);
		outputLayer[n].updateInputWeights(prevLayer);
	}
	//calc hidden gradients
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--) {
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < hiddenLayer.size() - 1; n++) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
			hiddenLayer[n].updateInputWeights(prevLayer);
		}
	}
}
void Net::feedForward(const vector<double> &inputVals) {
	assert(inputVals.size() == m_layers[0].size() - 1); //makes sure number of inputs and equal to number of input neurons
	//makes input neurons output a certain data input
	for (unsigned i = 0; i < inputVals.size(); i++) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}
	for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; n++) {
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
	vector<double> outputLayerVals;
	for (int s = 0; s < outputSections; s++) {
		outputLayerVals.clear();
		for (int p = s * (m_layers.back().size() - 1) / outputSections; p < (s + 1) * (m_layers.back().size() - 1) / outputSections; p++) {
			outputLayerVals.push_back(m_layers[2][p].getOutputVal());
		}
		for (int p = s * (m_layers.back().size() - 1) / outputSections; p < (s + 1) * (m_layers.back().size() - 1) / outputSections; p++) {
			m_layers[2][p].softMax(outputLayerVals);
		}
	}
}
Net::Net(const unsigned inputSize, const unsigned outputSec, const unsigned windowSize) {
	unsigned numLayers = 3;
	outputSections = outputSec;
	vector<unsigned> topology;
	topology.push_back(inputSize);
	topology.push_back(150);
	topology.push_back(inputSize / windowSize);
	n_recentAverageError = 1.0;
	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
		m_layers.push_back(Layer()); //adds a layer
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];//if layerNum is the output layer then set to 0 otherwise set to number of neurons in next layer
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			//cout << "added neuron for layer " << layerNum << endl;
		}
		cout << "Added " << topology[layerNum] + 1 << " neurons for layer " << layerNum << endl;
		m_layers.back().back().setOutputVal(rand() / double(RAND_MAX));
	}
}
/*
number of layers
neurons in layer 0 (includes bias)
neurons in layer 1
...
0 weights for neuron 0 in layer 0
1 ...
2 ...
...
b weights for bias neuron (final value is the bias output value)
0 weights for neuron 0 in layer 1
1 ...
...
...

*/
Net::Net(const string &fileName) {
	ifstream fin(fileName);
	int layerNum = 0;
	fin >> outputSections;
	fin >> layerNum;
	vector<unsigned> topology;
	for (int l = 0; l < layerNum; l++) {
		int num;
		fin >> num;
		topology.push_back(num);
	}
	for (int layer = 0; layer < layerNum; layer++) {
		m_layers.push_back(Layer());
		unsigned numOutputs = layer == topology.size() - 1 ? 0 : topology[layer + 1] - 1;
		for (int n = 0; n < topology[layer]; n++) {
			int index;
			fin >> index;
			vector<double> weights;
			for (int w = 0; w < numOutputs; w++) {
				double outW;
				fin >> outW;
				weights.push_back(outW);
			}
			m_layers.back().push_back(Neuron(weights, index));
			//cout << "add neuron for layer " << layer << endl;
			if (n == topology[layer] - 1) {
				double output = 0;
				fin >> output;
				m_layers.back().back().setOutputVal(output);
			}
		}
		cout << "Added " << topology[layer] + 1 << " neurons for layer " << layer << endl;
	}
}
void Net::storeNet(const string &fileName) {
	ofstream fout(fileName);
	fout << outputSections << endl;
	fout << m_layers.size() << endl;
	for (int l = 0; l < m_layers.size(); l++) {
		fout << m_layers[l].size() << endl;
	}
	for (int l = 0; l < m_layers.size(); l++) {
		for (int n = 0; n < m_layers[l].size(); n++) {
			fout << n << " ";
			vector<double> weights;
			m_layers[l][n].storeNeuron(weights);
			for (int w = 0; w < weights.size(); w++) {
				fout << weights[w] << " ";
			}
			if (n == m_layers[l].size() - 1) {
				fout << m_layers[l][n].getOutputVal();
			}
			fout << endl;
		}
	}
}