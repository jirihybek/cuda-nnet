/*
 * Spiking neural network
 *
 * Network generator
 */

#ifndef H_ARCHITECT
#define H_ARCHITECT

#include "network.h"

class Architect {
	public:
		static NNetwork *RecurrentFeedforward(int input_count, int hidden_layer_count, int hidden_neuron_count, int output_count);
};

#endif