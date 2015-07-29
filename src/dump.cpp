/*
 * Spiking neural network
 *
 * Network generator
 */

#include "dump.h"

std::string Dump::tree(NNetwork *net){

	std::string result = "";

	for(int l = 0; l < net->layers.size(); l++){

		//Print layer
		char b_lid[12];
		snprintf(b_lid, sizeof(b_lid), "Layer #%i\n", l);
		result.append(b_lid);

		//Print nodes
		for(int n = 0; n < net->layers.at(l)->nodes.size(); n++){

			NNode *node = net->layers.at(l)->nodes.at(n);

			char b_nid[255];
			
			if(node->type == 0){
				
				snprintf(b_nid, sizeof(b_nid), "    - Input #%i:%i | Value: %f \n", l, n, node->value);
				result.append(b_nid);

			} else {
				
				NNeuron *neuron = (NNeuron *) net->layers.at(l)->nodes.at(n);
				
				snprintf(b_nid, sizeof(b_nid), "    - Node #%i:%i | Threshold: %f | Sum: %f | Value: %f \n                |", l, n, neuron->threshold, neuron->sum, neuron->value);
				result.append(b_nid);

				for(int i = 0; i < neuron->inputs.size(); i++){

					char b_nii[12];
					snprintf(b_nii, sizeof(b_nii), " %i:%i:%f /", neuron->inputs.at(i)->layer_id, neuron->inputs.at(i)->node_id, neuron->inputs.at(i)->weight);
					result.append(b_nii);


				}

				result.append("\n");

			}
			

		}

		result.append("\n");

	}

	return result;

}

std::string Dump::json(NNetwork *net){

	std::string result = "[";

	for(int l = 0; l < net->layers.size(); l++){

		if(l > 0)
			result.append(",");

		result.append("[");

		//Print nodes
		for(int n = 0; n < net->layers.at(l)->nodes.size(); n++){

			if(n > 0)
				result.append(",");

			NNode *node = net->layers.at(l)->nodes.at(n);

			char b_nid[255];
			
			if(node->type == 0){
				
				snprintf(b_nid, sizeof(b_nid), "{\"type\":0,\"value\":%f}", node->value);
				result.append(b_nid);

			} else {
				
				NNeuron *neuron = (NNeuron *) net->layers.at(l)->nodes.at(n);
				
				snprintf(b_nid, sizeof(b_nid), "{\"type\":1,\"threshold\":%f,\"sum\":%f,\"value\":%f,\"inputs\":[", neuron->threshold, neuron->sum, neuron->value);
				result.append(b_nid);

				for(int i = 0; i < neuron->inputs.size(); i++){

					if(i > 0)
						result.append(",");

					char b_nii[64];
					snprintf(b_nii, sizeof(b_nii), "{\"layer\":%i,\"node\":%i,\"weight\":%f}", neuron->inputs.at(i)->layer_id, neuron->inputs.at(i)->node_id, neuron->inputs.at(i)->weight);
					result.append(b_nii);


				}

				result.append("]}");

			}
			

		}

		result.append("]");

	}

	result.append("]");

	return result;

}

std::string Dump::jsonState(NNetwork *net){

	std::string result = "[";

	for(int l = 0; l < net->layers.size(); l++){

		if(l > 0)
			result.append(",");

		result.append("[");

		//Print nodes
		for(int n = 0; n < net->layers.at(l)->nodes.size(); n++){

			if(n > 0)
				result.append(",");

			NNode *node = net->layers.at(l)->nodes.at(n);

			char b_nid[255];
			
			if(node->type == 0){
				
				snprintf(b_nid, sizeof(b_nid), "{\"v\":%f}", node->value);
				result.append(b_nid);

			} else {
				
				NNeuron *neuron = (NNeuron *) net->layers.at(l)->nodes.at(n);
				
				snprintf(b_nid, sizeof(b_nid), "{\"t\":%f,\"s\":%f,\"v\":%f,\"i\":[", neuron->threshold, neuron->sum, neuron->value);
				result.append(b_nid);

				for(int i = 0; i < neuron->inputs.size(); i++){

					if(i > 0)
						result.append(",");

					char b_nii[64];
					snprintf(b_nii, sizeof(b_nii), "{\"w\":%f}", neuron->inputs.at(i)->weight);
					result.append(b_nii);


				}

				result.append("]}");

			}
			

		}

		result.append("]");

	}

	result.append("]");

	return result;

}