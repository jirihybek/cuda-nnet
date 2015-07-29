.PHONY: build
.PHONY: clean
.PHONY: run
.PHONY: run_clean

BUILD   = build/main
OBJ_DIR = obj
SRC_DIR = src

HEADERS = network.h architect.h dump.h engine.h engine_cpu.h engine_gpu.cuh
OBJECTS = network.o architect.o dump.o engine.o engine_cpu.o engine_gpu.o main.o

OBJS = $(patsubst %,$(OBJ_DIR)/%,$(OBJECTS))
HDRS = $(patsubst %,$(SRC_DIR)/%,$(HEADERS))

build: ${OBJS}
	nvcc ${OBJS} -o ${BUILD}

run: build
	${BUILD} -i 100 -s 1 -j dump.js -t dump.tree 5 3 17 5

run_clean: clean run

clean:
	rm -f ${BUILD}
	rm -f ${OBJ_DIR}/*.o
	rm -f build/cuda_sum
	rm -f build/cuda_vector

${OBJ_DIR}/engine_gpu.o: ${SRC_DIR}/engine_gpu.cu ${HDRS}
	nvcc -c -o ${OBJ_DIR}/engine_gpu.o ${SRC_DIR}/engine_gpu.cu

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cpp ${HDRS}
	nvcc -c -o $@ $<
