/*
 * Spiking neural network
 *
 * Network dumper
 */

#ifndef H_DUMP
#define H_DUMP

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "network.h"

class Dump {
	public:
		static std::string tree(NNetwork *net);
		static std::string json(NNetwork *net);
		static std::string jsonState(NNetwork *net);
};

#endif