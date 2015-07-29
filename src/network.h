/*
 * Spiking neural network
 *
 * Network structure container
 */

#ifndef H_NETWORK
#define H_NETWORK

#include <vector>

using namespace std;

/*
 * Node interface class
 */
class NNode {
	public:
		float value;
		int type;
};

/*
 * Input
 */
class NInput : public NNode {
	public:
		NInput(float value);
};

/*
 * Neuron
 */
class NNeuronInput {
	public:	
		int layer_id;
		int node_id;
		float weight;
		NNeuronInput(int layer_id, int node_id, float weight);
};

class NNeuron : public NNode {
	public:
		float threshold;

		vector<NNeuronInput*> inputs;
	
		float sum;

		NNeuron(float threshold);
		~NNeuron();
		NNeuronInput *addInput(int layer_id, int node_id, float weight);
};

/*
 * Layer
 */
class NLayer {
	public:
		vector<NNode*> nodes;

		~NLayer();
		NInput *addInput(float value);
		NNeuron *addNeuron(float threshold);
};

/*
 * Network
 */
class NNetwork {
	public:
		vector<NLayer*> layers;
		~NNetwork();
		NLayer *addLayer();
		NLayer *getOutputLayer();
};

#endif