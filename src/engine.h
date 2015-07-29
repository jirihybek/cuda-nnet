/*
 * Spiking neural network
 *
 * Network engine base class
 */

#ifndef H_ENGINE
#define H_ENGINE

#include <vector>
#include "network.h"

class Engine {
	protected:
		NNetwork *net;
	public:
		Engine(NNetwork *net);
		virtual ~Engine() {}
		virtual void feed(std::vector<float> inputs) = 0;
		virtual void sync() = 0;
};

#endif