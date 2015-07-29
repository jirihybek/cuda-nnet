/*
 * Spiking neural network
 *
 * Network engine base class
 */

#ifndef H_ENGINE_GPU
#define H_ENGINE_GPU

#include "engine.h"

class Engine_GPU: public Engine {
	private:
		//Host memory
		int h_node_offset_last;
		int h_input_offset_last;

		int *h_input_layer;
		int *h_input_node;
		float *h_input_weight;

		float *h_node_threshold;
		float *h_node_sum;
		float *h_node_value;
		float *h_node_output;

		int *h_node_count;
		int *h_node_offset;

		int *h_input_count;
		int *h_input_offset;

		//Device memory
		int *d_input_layer;
		int *d_input_node;
		float *d_input_weight;

		float *d_node_threshold;
		float *d_node_sum;
		float *d_node_value;
		float *d_node_output;

		int *d_node_count;
		int *d_node_offset;

		int *d_input_count;
		int *d_input_offset;

		int getNodeIndex(int layer_id, int node_id);
		int getInputIndex(int node_index, int input_id);

		void copyToDevice();
		void copyToHost();

	public:
		Engine_GPU(NNetwork *net);
		~Engine_GPU();
		virtual void feed(std::vector<float> inputs);
		virtual void sync();
};

#endif