/*
 * Spiking neural network
 *
 * Network generator
 */

#include <time.h>
#include <stdlib.h>
#include "architect.h"

NNetwork *Architect::RecurrentFeedforward(int input_count, int hidden_layer_count, int hidden_neuron_count, int output_count){

	srand(time(NULL));

	NNetwork *net = new NNetwork();
	
	NLayer *layer_input = net->addLayer();
	
	NLayer *layer_last = layer_input;
	int layer_last_id = 0;

	//Add inputs
	for(int i = 0; i < input_count; i++){
		layer_input->addInput((rand() % 1000) / 500.0 - 1.0);
	}

	//Add input neurons
	NLayer *layer_input_neurons = net->addLayer();

	for(int n = 0; n < input_count; n++){

		NNeuron *neuron = layer_input_neurons->addNeuron(1);

		//Add inputs from previous layer
		for(int i = 0; i < layer_last->nodes.size(); i++)
			neuron->addInput(layer_last_id, i, rand() % 1000 / 500.0 - 1.0);

	}

	layer_last = layer_input_neurons;
	layer_last_id++;

	//Add hidden layers
	for(int l = 0; l < hidden_layer_count; l++){

		NLayer *layer_hidden = net->addLayer();

		//Add neurons
		for(int n = 0; n < hidden_neuron_count; n++){

			NNeuron *neuron = layer_hidden->addNeuron(rand() % 1000 / 1000.0 + 1.0);

			//Add inputs from previous layer
			//for(int i = 0; i < layer_last->nodes.size(); i++)
			//	neuron->addInput(layer_last_id, i, rand() % 1000 / 500.0 - 1.0);

		}

		layer_last = layer_hidden;
		layer_last_id++;

	}

	//Add output layer
	NLayer *layer_output = net->addLayer();

	//Add neurons
	for(int n = 0; n < output_count; n++){

		NNeuron *neuron = layer_output->addNeuron(1);

		//Add inputs from previous layer
		for(int i = 0; i < layer_last->nodes.size(); i++)
			neuron->addInput(layer_last_id, i, rand() % 1000 / 500.0 - 1.0);

	}

	//Make recurrent connections
	for(int l = 1; l < net->layers.size() - 1; l++){
		for(int n = 0; n < net->layers.at(l)->nodes.size(); n++){
			for(int i = 0; i < (int) hidden_neuron_count / 2; i++){

				int target_layer, target_node;

				//Get layer
				if(rand() % 3 == 0){
					//Other layer
					target_layer = (int) rand() % (net->layers.size() - 2) + 1;

				} else {
					//Same layer
					target_layer = l;

				}

				//Get target node
				target_node = (int) rand() % net->layers.at(target_layer)->nodes.size();

				if(target_layer == l && target_node == n) continue;

				NNeuron *neuron = (NNeuron *)net->layers.at(l)->nodes.at(n);

				bool cnt = true;
				for(int c = 0; c < neuron->inputs.size(); c++)
					if(neuron->inputs.at(c)->layer_id == target_layer && neuron->inputs.at(c)->node_id == target_node){
						cnt = false;
						break;
					}

				if(!cnt) continue;

				neuron->addInput(target_layer, target_node, rand() % 1000 / 500.0 - 1.0);				

			}
		}
	}

	return net;

}
