/*
 * Spiking neural network
 *
 * Network engine base class
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <signal.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#include "architect.h"
#include "dump.h"
#include "engine.h"
#include "engine_cpu.h"
#include "engine_gpu.cuh"

volatile sig_atomic_t flag = 0;
bool isInterrupted = false;

int usage(){

	printf("Usage: main [options] <input count> <hidden layer count> <hidden layer neurons> <output count>\n");
	printf("\n");
	printf("  -i <count>     Count of iterations (if not specified infinite loop is started)\n");
	printf("  -s <count>     Make output every STEP (if not specified, one output is pefromed at the end)\n");
	printf("  -j <filename>  JS output filename\n");
	printf("  -t <filename>  Tree output filename\n");
	printf("  -c             Use CPU engine\n");
	printf("\n");

	return 1;

}

void handleInterrupt(int sig){
	isInterrupted = true;
}

int main(int argc, char** argv){

	bool useCpu = false;
	bool dumpJs = false;
	bool dumpTree = false;

	std::ofstream dumpJs_stream;
	std::ofstream dumpTree_stream;

	int iterCount = 0;
	int dumpStep = 0;

	int stepCounter = 0;

	int netConfig[4];
	int netConfigIndex = 0;

	time_t timeStart;
	time_t timeElapsed;

	//Parse arguments
	for(int a = 1; a < argc; a++){
		if(strncmp(argv[a], "-", 1) == 0){
			
			switch(argv[a][1]){
				case 'i':
					iterCount = atoi(argv[a + 1]);
					a++;
					break;

				case 's':
					dumpStep = atoi(argv[a + 1]);
					a++;
					break;

				case 'j':
					dumpJs_stream.open(argv[a + 1]);
					dumpJs = true;
					a++;
					break;

				case 't':
					dumpTree_stream.open(argv[a + 1]);
					dumpTree = true;
					a++;
					break;

				case 'c':
					useCpu = true;
					break;

				default:
					printf("Unknown option: -%c\n\n", argv[a][1]);
					return usage();
			}

		} else {
			
			if(netConfigIndex > 3) continue;

			netConfig[netConfigIndex] = atoi(argv[a]);
			netConfigIndex++;

		}

	}

	if(netConfigIndex < 4){
		printf("To few network config params\n\n");
		return usage();
	}

	/* CONFIG DUMP
	printf("Use CPU:         %s\n", ( useCpu ? "YES": "no" ));
	printf("Dump JSON:       %s\n", ( dumpJs ? "YES": "no" ));
	printf("Dump Tree:       %s\n", ( dumpTree ? "YES": "no" ));
	printf("Iteration count: %d\n", iterCount);
	printf("Dump per steps:  %d\n", dumpStep);
	printf("Network config:  %d / %d / %d / %d\n\n", netConfig[0], netConfig[1], netConfig[2], netConfig[3]);
	*/

	//Create network
	printf("Creating network...\n  - Input count:          %d\n  - Hidden layer count:   %d\n  - Hidden layer neurons: %d\n  - Output count:         %d\n", netConfig[0], netConfig[1], netConfig[2], netConfig[3]);
	NNetwork *net = Architect::RecurrentFeedforward(netConfig[0], netConfig[1], netConfig[2], netConfig[3]);

	//Create engine
	printf("Creating engine: ");

	Engine *engine;
	int cudaDeviceCount;

	cudaGetDeviceCount(&cudaDeviceCount);

	if( !useCpu && cudaDeviceCount > 0 ){
		printf("CUDA\n");
		engine = new Engine_GPU(net);
	} else {
		printf("CPU\n");
		engine = new Engine_CPU(net);
	}

	//Declare inputs
	vector<float> inputs(5);
	srand(time(NULL));

	printf("Starting simulation...\n");

	//Write json dump?
	if(dumpJs)
		dumpJs_stream << "var network = " << Dump::json(net) << "; var states = [" << Dump::jsonState(net);

	//Register signal
	signal(SIGINT, handleInterrupt);

	if(iterCount == 0)
		printf("Stop by pressing Ctrl+C\n");

	//Set start time
	timeStart = time(NULL);

	//Start loop
	while(!isInterrupted && (iterCount == 0 || stepCounter < iterCount)){

		//Set inputs
		inputs[0] = rand() % 1000 / 1000.0;
		inputs[1] = rand() % 1000 / 1000.0;
		inputs[2] = rand() % 1000 / 1000.0;
		inputs[3] = rand() % 1000 / 1000.0;
		inputs[4] = rand() % 1000 / 1000.0;

		//Compute
		engine->feed(inputs);

		//Dump?
		if(dumpStep > 0 && stepCounter % dumpStep == 0){

			engine->sync();

			if(dumpJs)
				dumpJs_stream << "," << Dump::jsonState(net);
			
			if(dumpTree)
				dumpTree_stream << Dump::tree(net) << std::endl;

		}

		stepCounter++;
		
	}

	//End time
	timeElapsed = difftime(time(NULL), timeStart);

	//Sync
	engine->sync();

	printf("Stopping...\n");
	printf("Total time: %i seconds\n", (int) timeElapsed);

	//Final dump and close dump streams
	if(dumpJs){
		dumpJs_stream << "," << Dump::jsonState(net) << "];";
		dumpJs_stream.close();
	}

	if(dumpTree){
		dumpTree_stream << Dump::tree(net);
		dumpTree_stream.close();
	}

	//Free memory
	delete engine;
	delete net;

	return 0;

}