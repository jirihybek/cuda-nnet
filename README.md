# cuda-nnet
Spiking neural network implementation using CPU and CUDA engine

Demonstration screencast:  
[https://www.youtube.com/watch?v=Oe1ldwEEwsI](https://www.youtube.com/watch?v=Oe1ldwEEwsI)

## Requirements
 - g++ compiler
 - nVidia CUDA Toolkit (nvcc compiler, cuda headers)
 - make

## Compilation
```
make            # Build
make clean      # Clean binaries
make run        # Build and run
make run_clean  # Clean, build and run
```

Tested:
 - without CUDA on Macbook Pro
 - with CUDA on Ubuntu 14.04

Not tested in MS Windows environment. Usage of CygWin or another POSIX environment is recommended.

## Usage
```
Usage: main [options] <input count> <hidden layer count> <hidden layer neurons> <output count>

  -i <count>     Count of iterations (if not specified infinite loop is started)
  -s <count>     Make output every STEP (if not specified, one output is pefromed at the end)
  -j <filename>  JS output filename
  -t <filename>  Tree output filename
  -c             Use CPU engine
```

### Examples
```
# Create network with 3 inputs, 1 hidden layer with 5 neurons and 2 outputs. No dump and infinite iterations until user interrupt.
./main 3 1 5 2

# Create network same as above but do only 100 iterations.
./main -i 100 3 1 5 2

# Create network same as above do 100 iterations and dump to JSON file every 10th step.
./main -i 100 -s 10 -j dump.js 3 1 5 2

# Same as above but dump every step.
./main -i 100 -s 1 -j dump.js 3 1 5 2
```

## Viewer
Project contains HTML viewer. To use viewer run application with following options:

```
main -s 1 -j dump.js
```

It will write network structure and state of every iteration into `dump.js` file.

Then open `viewer.html` in browser (use Google Chrome or another HTML5 compliant browser) and enter dump file name.

Then you can browse network step by step or play as animation.

# Technical details

## Files description

```
architect.cpp      # Helps with creating network structure (creates nodes and connections)
dump.cpp           # Functions for dumping network into JSON and readable tree-like format
engine.cpp         # Base class for computation engines
engine_cpu.cpp     # CPU engine
engine_gpu.cu      # CUDA GPU engine
main.cpp           # Application main file
network.cpp        # Network structure and state container
```

## Network
Network consists of **layers**.

Each **layer** consists of **inputs** and **neurons** which are **nodes**.

Each **node** has **value**.

Each **neuron** has **threshold**, **sum** (action potential) and **inputs**.

Each **input** has **weight** and target node described by **layer_id** and **node_id**.

## Engines
Engines are created for already defined networks. They provides **feed** and **sync** methods.

Method **feed** sets network inputs and calculates a step.  

Method **sync** synchronizes current network state (which is in engine memory) with network structure (which can be dumped).

## CPU engine
CPU engine is demonstration of CUDA principle but computed using CPU.

Code looks ugly but one thing needs to be done. It is the flattening. Because GPU has better performance using 1D arrays.

So complex network structure needs to be flattened to one dimension which can be then effectively processed.

## GPU CUDA engine
GPU engine uses the same principle as CPU engine.

GPU engine computes network in parallel:
 - Each layer is computed by block
 - Each node in layer is computed by thread

GPU engine copies network structure into memory once and then only updates inputs.

## Performance observations
 - CPU and GPU engines has same performance with small networks and many iterations.
 - GPU has better performance with larger networks.

## References
 - [Flattening algorithm for jagged arrays](http://stackoverflow.com/questions/31662370/2d-jagged-array-to-1d-array-in-c/31662573) (thanks to [Robert Crovella](http://stackoverflow.com/users/1695960/robert-crovella) from nVidia)
 - [nVidia CUDA tutorial](http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf)

## Possible improvements
 - Apply DRY principle (Don't Repeat Yourself) to engines
 - Architect methods for creating different ANN models
 - Extension of engines to support classical feed-forward computation with activation functions and backpropagation learning
 - OpenCL engine