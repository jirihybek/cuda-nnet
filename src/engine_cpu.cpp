/*
 * Spiking neural network
 *
 * Network generator
 */

#include <stdlib.h>
#include <algorithm>

#include "engine_cpu.h"

Engine_CPU::Engine_CPU(NNetwork *net) : Engine(net) {
	
	int node_offset_last = 0;
	int input_offset_last = 0;

	//Set node offsets
	this->node_offset = (int *) malloc(sizeof(int) * net->layers.size());
	this->node_count = (int *) malloc(sizeof(int) * net->layers.size());

	for(int i = 0; i < net->layers.size(); i++){
		this->node_count[i] = net->layers.at(i)->nodes.size();
		this->node_offset[i] = node_offset_last;
		node_offset_last+= net->layers.at(i)->nodes.size();
	}

	//Set node data and set input offsets
	this->node_threshold = (float *) malloc(sizeof(float) * node_offset_last);
	this->node_sum = (float *) malloc(sizeof(float) * node_offset_last);
	this->node_value = (float *) malloc(sizeof(float) * node_offset_last);
	this->node_output = (float *) malloc(sizeof(float) * node_offset_last);

	this->input_offset = (int *) malloc(sizeof(int) * (node_offset_last + 1));
	this->input_count = (int *) malloc(sizeof(int) * (node_offset_last + 1));

	for(int l = 0; l < net->layers.size(); l++){
		for(int n = 0; n < net->layers.at(l)->nodes.size(); n++){

			NNode *node = net->layers.at(l)->nodes.at(n);
			int node_index = this->getNodeIndex(l, n);

			if(node->type == 0){

				this->node_threshold[node_index] = 0;
				this->node_sum[node_index] = 0;
				this->node_value[node_index] = node->value;
				this->node_output[node_index] = node->value;

				this->input_offset[node_index] = input_offset_last;
				this->input_count[node_index] = 0;

			} else {

				NNeuron *neuron = (NNeuron *) net->layers.at(l)->nodes.at(n);

				this->node_threshold[node_index] = neuron->threshold;
				this->node_sum[node_index] = neuron->sum;
				this->node_value[node_index] = neuron->value;
				this->node_output[node_index] = neuron->value;

				this->input_offset[node_index] = input_offset_last;
				this->input_count[node_index] = neuron->inputs.size();
				input_offset_last+= neuron->inputs.size();

			}

		}
	}

	this->input_offset[node_offset_last] = input_offset_last;

	//Set inputs
	this->input_layer = (int *) malloc(sizeof(int) * input_offset_last);
	this->input_node = (int *) malloc(sizeof(int) * input_offset_last);
	this->input_weight = (float *) malloc(sizeof(float) * input_offset_last);

	for(int l = 0; l < net->layers.size(); l++){
		for(int n = 0; n < net->layers.at(l)->nodes.size(); n++){

			if(net->layers.at(l)->nodes.at(n)->type == 1){

				int node_index = this->getNodeIndex(l, n);
				NNeuron *neuron = (NNeuron *) net->layers.at(l)->nodes.at(n);

				for(int i = 0; i < neuron->inputs.size(); i++){
					
					int input_index = this->getInputIndex(node_index, i);

					this->input_layer[input_index] = neuron->inputs.at(i)->layer_id;
					this->input_node[input_index] = neuron->inputs.at(i)->node_id;
					this->input_weight[input_index] = neuron->inputs.at(i)->weight;

				}

			}

		}
	}

}

Engine_CPU::~Engine_CPU(){
	free(this->node_count);
	free(this->node_offset);

	free(this->input_count);
	free(this->input_offset);

	free(this->node_threshold);
	free(this->node_sum);
	free(this->node_value);
	free(this->node_output);

	free(this->input_layer);
	free(this->input_node);
	free(this->input_weight);
}

int Engine_CPU::getNodeIndex(int layer_id, int node_id){
	return this->node_offset[layer_id] + node_id;
}

int Engine_CPU::getInputIndex(int node_index, int input_id){
	return this->input_offset[node_index] + input_id;
}

void Engine_CPU::kernel(int block_id, int thread_id){

	if(thread_id >= this->node_count[block_id])
		return;

	//Get indexes
	int node_index = this->getNodeIndex(block_id, thread_id);
	int input_index_begin = this->input_offset[node_index];
	int input_index_end = this->input_offset[node_index + 1];

	//Flop values (double-buffer)
	this->node_value[node_index] = this->node_output[node_index];
	this->node_output[node_index] = 0;

	//Make sum of current values
	float sum = this->node_sum[node_index];

	for(int i = input_index_begin; i < input_index_end; i++){
		
		int target_index = this->getNodeIndex( this->input_layer[i], this->input_node[i] );

		sum+= this->node_value[target_index] * this->input_weight[i];

	}

	if(sum > this->node_threshold[node_index]){
		this->node_sum[node_index] = 0;
		this->node_output[node_index] = 1;
	} else {
		this->node_sum[node_index] = sum;
	}

}

void Engine_CPU::feed(std::vector<float> inputs){

	int blocks = (int) this->net->layers.size();
	int threads = 0;

	for(int i = 0; i < this->net->layers.size(); i++)
		threads = std::max(threads, (int) this->net->layers.at(i)->nodes.size());

	//Set inputs into memory
	for(int i = 0; i < inputs.size(); i++)
		this->node_value[this->getNodeIndex(0, i)] = inputs.at(i);

	//Run kernel
	for(int b = 1; b < blocks; b++)
		for(int t = 0; t < threads; t++)
			this->kernel(b, t);

}

void Engine_CPU::sync(){

	for(int l = 0; l < net->layers.size(); l++){
		for(int n = 0; n < net->layers.at(l)->nodes.size(); n++){

			NNode *node = net->layers.at(l)->nodes.at(n);
			int node_index = this->getNodeIndex(l, n);

			if(node->type == 0){

				node->value = this->node_value[node_index];

			} else {

				NNeuron *neuron = (NNeuron *) net->layers.at(l)->nodes.at(n);

				neuron->threshold = this->node_threshold[node_index];
				neuron->sum = this->node_sum[node_index];
				neuron->value = this->node_value[node_index];

				for(int i = 0; i < neuron->inputs.size(); i++){
					
					int input_index = this->getInputIndex(node_index, i);
					neuron->inputs.at(i)->weight = this->input_weight[input_index];

				}

			}

		}
	}

}