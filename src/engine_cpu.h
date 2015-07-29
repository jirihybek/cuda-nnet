/*
 * Spiking neural network
 *
 * Network engine base class
 */

#ifndef H_ENGINE_CPU
#define H_ENGINE_CPU

#include "engine.h"

class Engine_CPU: public Engine {
	private:
		int *input_layer;
		int *input_node;
		float *input_weight;

		float *node_threshold;
		float *node_sum;
		float *node_value;
		float *node_output;

		int *node_count;
		int *node_offset;

		int *input_count;
		int *input_offset;

		int getNodeIndex(int layer_id, int node_id);
		int getInputIndex(int node_index, int input_id);

		void kernel(int block_id, int thread_id);

	public:
		Engine_CPU(NNetwork *net);
		~Engine_CPU();
		virtual void feed(std::vector<float> inputs);
		virtual void sync();
};

#endif