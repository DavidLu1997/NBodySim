//N-Body Physics Simulation
//Randomly generates and displays an N-Body system
//Code from Nvidia's GPU Gems 3 Chapter 31

#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

//N, number of bodies
const unsigned int N = 100;

//p, number of blocks
const unsigned int p = 5;

//Interaction between two bodies
//Integrate using leapfrog-Verlet integrator
//a ~= G * summation(1, N, m[j]*r[i][j] / (r^2 + E^2)^3/2
//G = 1 for simplicity
__device__ float3 bobyBodyInteraction(float4 b1, float4 b2, float3 a1) {
	//Distance between bodies
	float3 d;
	//3 FLOPS
	d.x = b2.x - b1.x;
	d.y = b2.y - b1.y;
	d.z = b2.z - b1.z;

	//Square distance
	//6 FLOPS
	float square = d.x * d.x + d.y * d.y + d.z * d.z;

	//Cube
	//2 FLOPS
	float cube = square * square * square;

	//Invert and sqrt
	//2 FLOPS
	float invert = 1.0f / sqrtf(cube);

	//Calculate s
	float s = b2.w * invert;


	//Calculate a
	a1.x += d.x * s;
	a1.y += d.y * s;
	a1.z += d.z * s;
	return a1;

}

//Calculate all interactions within a tile
__device__ float3 tile_calculation(float4 pos, float3 a) {
	int i;
	//Used shared memory
	extern __shared__ float4 sPos[];
	for (i = 0; i < blockDim.x; i++) {
		a = bobyBodyInteraction(pos, sPos[i], a);
	}
	return a;
}

//Calculate acceleration for p bodies with p threads resulting from N interactions
__global__ void calculate_forces(void *devX, void *devA) {
	//Declare shared position
	extern __shared__ float4 sPos[];

	//Get from memory
	float4 *globalX = (float4 *)devX;
	float4 *globalA = (float4 *)devA;

	//Initialize position
	float4 pos;

	//Initialize variables
	int i, tile;
	float3 acc = { 0.0f, 0.0f, 0.0f };

	//Get index
	int getId = blockIdx.x * blockDim.x + threadIdx.x;

	//Get position
	pos = globalX[getId];

	//Calculate N bodies
	for (i = 0, tile = 0; i < N; i += p, tile++) {
		//Get index
		int idx = tile * blockDim.x + threadIdx.x;

		//Update position
		sPos[threadIdx.x] = globalX[idx];

		//Barrier
		__syncthreads();

		//Get acceleration
		acc = tile_calculation(pos, acc);

		//Barrier
		__syncthreads();
	}

	//Save in global memory
	float4 acc4 = { acc.x, acc.y, acc.z, 0.0f };
	globalA[getId] = acc4;

}

int main()
{

    return 0;
}