/*
 * Spiking neural network
 *
 * Network generator
 */

#include <stdlib.h>
#include <stdio.h>
#include <algorithm>

#include "engine_gpu.cuh"

Engine_GPU::Engine_GPU(NNetwork *net) : Engine(net) {
	
	//Set node offsets
	this->h_node_offset = (int *) malloc(sizeof(int) * net->layers.size());
	this->h_node_count = (int *) malloc(sizeof(int) * net->layers.size());

	for(int i = 0; i < net->layers.size(); i++){
		this->h_node_count[i] = net->layers.at(i)->nodes.size();
		this->h_node_offset[i] = this->h_node_offset_last;
		this->h_node_offset_last+= net->layers.at(i)->nodes.size();
	}

	//Set node data and set input offsets
	this->h_node_threshold = (float *) malloc(sizeof(float) * this->h_node_offset_last);
	this->h_node_sum = (float *) malloc(sizeof(float) * this->h_node_offset_last);
	this->h_node_value = (float *) malloc(sizeof(float) * this->h_node_offset_last);
	this->h_node_output = (float *) malloc(sizeof(float) * this->h_node_offset_last);

	this->h_input_offset = (int *) malloc(sizeof(int) * (this->h_node_offset_last + 1));
	this->h_input_count = (int *) malloc(sizeof(int) * (this->h_node_offset_last + 1));

	for(int l = 0; l < net->layers.size(); l++){
		for(int n = 0; n < net->layers.at(l)->nodes.size(); n++){

			NNode *node = net->layers.at(l)->nodes.at(n);
			int node_index = this->getNodeIndex(l, n);

			if(node->type == 0){

				this->h_node_threshold[node_index] = 0;
				this->h_node_sum[node_index] = 0;
				this->h_node_value[node_index] = node->value;
				this->h_node_output[node_index] = node->value;

				this->h_input_offset[node_index] = this->h_input_offset_last;
				this->h_input_count[node_index] = 0;

			} else {

				NNeuron *neuron = (NNeuron *) net->layers.at(l)->nodes.at(n);

				this->h_node_threshold[node_index] = neuron->threshold;
				this->h_node_sum[node_index] = neuron->sum;
				this->h_node_value[node_index] = neuron->value;
				this->h_node_output[node_index] = neuron->value;

				this->h_input_offset[node_index] = this->h_input_offset_last;
				this->h_input_count[node_index] = neuron->inputs.size();
				this->h_input_offset_last+= neuron->inputs.size();

			}

		}
	}

	this->h_input_offset[this->h_node_offset_last] = this->h_input_offset_last;

	//Set inputs
	this->h_input_layer = (int *) malloc(sizeof(int) * this->h_input_offset_last);
	this->h_input_node = (int *) malloc(sizeof(int) * this->h_input_offset_last);
	this->h_input_weight = (float *) malloc(sizeof(float) * this->h_input_offset_last);

	for(int l = 0; l < net->layers.size(); l++){
		for(int n = 0; n < net->layers.at(l)->nodes.size(); n++){

			if(net->layers.at(l)->nodes.at(n)->type == 1){

				int node_index = this->getNodeIndex(l, n);
				NNeuron *neuron = (NNeuron *) net->layers.at(l)->nodes.at(n);

				for(int i = 0; i < neuron->inputs.size(); i++){
					
					int input_index = this->getInputIndex(node_index, i);

					this->h_input_layer[input_index] = neuron->inputs.at(i)->layer_id;
					this->h_input_node[input_index] = neuron->inputs.at(i)->node_id;
					this->h_input_weight[input_index] = neuron->inputs.at(i)->weight;

				}

			}

		}
	}

	//Allocate device memory and copy to it
	cudaMalloc((void **)&this->d_node_count, sizeof(int) * net->layers.size());
	cudaMalloc((void **)&this->d_node_offset, sizeof(int) * net->layers.size());

	cudaMalloc((void **)&this->d_input_count, sizeof(int) * (this->h_node_offset_last + 1));
	cudaMalloc((void **)&this->d_input_offset, sizeof(int) * (this->h_node_offset_last + 1));

	cudaMalloc((void **)&this->d_node_threshold, sizeof(float) * this->h_node_offset_last);
	cudaMalloc((void **)&this->d_node_sum, sizeof(float) * this->h_node_offset_last);
	cudaMalloc((void **)&this->d_node_value, sizeof(float) * this->h_node_offset_last);
	cudaMalloc((void **)&this->d_node_output, sizeof(float) * this->h_node_offset_last);

	cudaMalloc((void **)&this->d_input_layer, sizeof(int) * this->h_input_offset_last);
	cudaMalloc((void **)&this->d_input_node, sizeof(int) * this->h_input_offset_last);
	cudaMalloc((void **)&this->d_input_weight, sizeof(float) * this->h_input_offset_last);

	//Copy to device
	this->copyToDevice();

}

Engine_GPU::~Engine_GPU(){
	//Free cuda memory
	cudaFree(this->d_node_count);
	cudaFree(this->d_node_offset);

	cudaFree(this->d_input_count);
	cudaFree(this->d_input_offset);

	cudaFree(this->d_node_threshold);
	cudaFree(this->d_node_sum);
	cudaFree(this->d_node_value);
	cudaFree(this->d_node_output);

	cudaFree(this->d_input_layer);
	cudaFree(this->d_input_node);
	cudaFree(this->d_input_weight);

	//Free host memory
	free(this->h_node_count);
	free(this->h_node_offset);

	free(this->h_input_count);
	free(this->h_input_offset);

	free(this->h_node_threshold);
	free(this->h_node_sum);
	free(this->h_node_value);
	free(this->h_node_output);

	free(this->h_input_layer);
	free(this->h_input_node);
	free(this->h_input_weight);
}

int Engine_GPU::getNodeIndex(int layer_id, int node_id){
	return this->h_node_offset[layer_id] + node_id;
}

int Engine_GPU::getInputIndex(int node_index, int input_id){
	return this->h_input_offset[node_index] + input_id;
}

void Engine_GPU::copyToDevice(){
	cudaMemcpy(this->d_node_count, this->h_node_count, sizeof(int) * this->net->layers.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_node_offset, this->h_node_offset, sizeof(int) * this->net->layers.size(), cudaMemcpyHostToDevice);

	cudaMemcpy(this->d_input_count, this->h_input_count, sizeof(int) * (this->h_node_offset_last + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_input_offset, this->h_input_offset, sizeof(int) * (this->h_node_offset_last + 1), cudaMemcpyHostToDevice);

	cudaMemcpy(this->d_node_threshold, this->h_node_threshold, sizeof(float) * this->h_node_offset_last, cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_node_sum, this->h_node_sum, sizeof(float) * this->h_node_offset_last, cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_node_value, this->h_node_value, sizeof(float) * this->h_node_offset_last, cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_node_output, this->h_node_output, sizeof(float) * this->h_node_offset_last, cudaMemcpyHostToDevice);

	cudaMemcpy(this->d_input_layer, this->h_input_layer, sizeof(int) * this->h_input_offset_last, cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_input_node, this->h_input_node, sizeof(int) * this->h_input_offset_last, cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_input_weight, this->h_input_weight, sizeof(float) * this->h_input_offset_last, cudaMemcpyHostToDevice);
};

void Engine_GPU::copyToHost(){
	cudaMemcpy(this->h_node_count, this->d_node_count, sizeof(int) * this->net->layers.size(), cudaMemcpyDeviceToHost);
	cudaMemcpy(this->h_node_offset, this->d_node_offset, sizeof(int) * this->net->layers.size(), cudaMemcpyDeviceToHost);

	cudaMemcpy(this->h_input_count, this->d_input_count, sizeof(int) * (this->h_node_offset_last + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(this->h_input_offset, this->d_input_offset, sizeof(int) * (this->h_node_offset_last + 1), cudaMemcpyDeviceToHost);

	cudaMemcpy(this->h_node_threshold, this->d_node_threshold, sizeof(float) * this->h_node_offset_last, cudaMemcpyDeviceToHost);
	cudaMemcpy(this->h_node_sum, this->d_node_sum, sizeof(float) * this->h_node_offset_last, cudaMemcpyDeviceToHost);
	cudaMemcpy(this->h_node_value, this->d_node_value, sizeof(float) * this->h_node_offset_last, cudaMemcpyDeviceToHost);
	cudaMemcpy(this->h_node_output, this->d_node_output, sizeof(float) * this->h_node_offset_last, cudaMemcpyDeviceToHost);

	cudaMemcpy(this->h_input_layer, this->d_input_layer, sizeof(int) * this->h_input_offset_last, cudaMemcpyDeviceToHost);
	cudaMemcpy(this->h_input_node, this->d_input_node, sizeof(int) * this->h_input_offset_last, cudaMemcpyDeviceToHost);
	cudaMemcpy(this->h_input_weight, this->d_input_weight, sizeof(float) * this->h_input_offset_last, cudaMemcpyDeviceToHost);
};

__global__ void Engine_GPU_flop(float *d_node_value, float *d_node_output, int *d_node_count, int *d_node_offset) {
	
	if(blockIdx.x > 0){

		if(threadIdx.x < d_node_count[blockIdx.x]){
			
			//Get indexes
			int node_index = d_node_offset[blockIdx.x] + threadIdx.x;
			
			//Flop values (double-buffer)
			d_node_value[node_index] = d_node_output[node_index];
			d_node_output[node_index] = 0;

		}
	}

}

__global__ void Engine_GPU_kernel(int *d_input_layer, int *d_input_node, float *d_input_weight, float *d_node_threshold, float *d_node_sum, float *d_node_value, float *d_node_output, int *d_node_count, int *d_node_offset, int *d_input_count, int *d_input_offset) {

	if(blockIdx.x > 0){

		if(threadIdx.x < d_node_count[blockIdx.x]){
			
			//Get indexes
			int node_index = d_node_offset[blockIdx.x] + threadIdx.x;
			int input_index_begin = d_input_offset[node_index];
			int input_index_end = d_input_offset[node_index + 1];
			
			//Flop values (double-buffer)
			//d_node_value[node_index] = d_node_output[node_index];
			//d_node_output[node_index] = 0;
			//NOTE: moved to separated kernel because of data race

			//Make sum of current values
			float sum = d_node_sum[node_index];

			for(int i = input_index_begin; i < input_index_end; i++){
				
				int target_index = d_node_offset[d_input_layer[i]] + d_input_node[i];

				sum+= d_node_value[target_index] * d_input_weight[i];

			}
			
			if(sum > d_node_threshold[node_index]){
				d_node_sum[node_index] = 0;
				d_node_output[node_index] = 1;
			} else {
				d_node_sum[node_index] = sum;
			}

		}

	}

}

void Engine_GPU::feed(std::vector<float> inputs){

	int blocks = (int) this->net->layers.size();
	int threads = 0;

	for(int i = 0; i < this->net->layers.size(); i++)
		threads = std::max(threads, (int) this->net->layers.at(i)->nodes.size());

	//Set inputs into memory
	for(int i = 0; i < inputs.size(); i++)
		this->h_node_value[this->getNodeIndex(0, i)] = inputs.at(i);

	//Copy inputs to device
	cudaMemcpy(this->d_node_value, this->h_node_value, sizeof(float) * this->h_node_offset_last, cudaMemcpyHostToDevice);

	//Run kernel
	dim3 BlockDim = dim3(blocks, 1, 1);
    dim3 ThreadDim  = dim3(threads, 1, 1);

	Engine_GPU_kernel<<<BlockDim,ThreadDim>>>(this->d_input_layer, this->d_input_node, this->d_input_weight, this->d_node_threshold, this->d_node_sum, this->d_node_value, this->d_node_output, this->d_node_count, this->d_node_offset, this->d_input_count, this->d_input_offset);

	cudaDeviceSynchronize();

	Engine_GPU_flop<<<BlockDim,ThreadDim>>>(this->d_node_value, this->d_node_output, this->d_node_count, this->d_node_offset);

	//printf("ErrP: %s\n", cudaGetErrorString(cudaPeekAtLastError()) );

}

void Engine_GPU::sync(){

	this->copyToHost();

	for(int l = 0; l < net->layers.size(); l++){
		for(int n = 0; n < net->layers.at(l)->nodes.size(); n++){

			NNode *node = net->layers.at(l)->nodes.at(n);
			int node_index = this->getNodeIndex(l, n);

			if(node->type == 0){

				node->value = this->h_node_value[node_index];

			} else {

				NNeuron *neuron = (NNeuron *) net->layers.at(l)->nodes.at(n);

				neuron->threshold = this->h_node_threshold[node_index];
				neuron->sum = this->h_node_sum[node_index];
				neuron->value = this->h_node_value[node_index];

				for(int i = 0; i < neuron->inputs.size(); i++){
					
					int input_index = this->getInputIndex(node_index, i);
					neuron->inputs.at(i)->weight = this->h_input_weight[input_index];

				}

			}

		}
	}

}