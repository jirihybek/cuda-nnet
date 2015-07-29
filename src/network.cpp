/*
 * Spiking neural network
 *
 * Network structure container
 */

#include "network.h"

/*
 * Input
 */
NInput::NInput(float value){
	this->value = value;
	this->type = 0;
}

/*
 * Neuron
 */
NNeuronInput::NNeuronInput(int layer_id, int node_id, float weight){
	this->layer_id = layer_id;
	this->node_id = node_id;
	this->weight = weight;
}

NNeuron::NNeuron(float threshold){
	this->type = 1;
	this->sum = 0;
	this->value = 0;
	this->threshold = threshold;
}

NNeuron::~NNeuron(){
	for(int i = 0; i < this->inputs.size(); i++)
		delete this->inputs.at(i);
}

NNeuronInput *NNeuron::addInput(int layer_id, int node_id, float weight){
	NNeuronInput *input = new NNeuronInput(layer_id, node_id, weight);
	this->inputs.push_back(input);

	return input;
}

/*
 * Layer
 */
NLayer::~NLayer(){
	for(int i = 0; i < this->nodes.size(); i++)
		delete this->nodes.at(i);
}

NInput *NLayer::addInput(float value){
	NInput *node = new NInput(value);
	this->nodes.push_back(node);

	return node;
}

NNeuron *NLayer::addNeuron(float threshold){
	NNeuron *node = new NNeuron(threshold);
	this->nodes.push_back(node);

	return node;
}

/*
 * Network
 */
NLayer *NNetwork::addLayer(){
	NLayer *layer = new NLayer();
	this->layers.push_back(layer);

	return layer;
}

NNetwork::~NNetwork(){
	for(int i = 0; i < this->layers.size(); i++)
		delete this->layers.at(i);
}

NLayer *NNetwork::getOutputLayer(){
	return this->layers.at( this->layers.size() - 1 );
}