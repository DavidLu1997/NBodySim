//N-Body Physics Simulation
//Randomly generates and displays an N-Body system
//Code from Nvidia's GPU Gems 3 Chapter 31

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <ctime>

//Gravitational constant
#define G 1

//N, number of bodies
const unsigned int N = 100;

//p, number of blocks
const unsigned int p = 5;

//Upper bounds for location
const float upperX = 100.0;
const float upperY = 100.0;
const float upperZ = 100.0;

//Lower bounds for location
const float lowerX = -100.0;
const float lowerY = -100.0;
const float lowerZ = -100.0;

//Mass bounds
const float upperMass = 100.0;
const float lowerMass = 1.0;

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
	a1.x += d.x * s * G;
	a1.y += d.y * s * G;
	a1.z += d.z * s * G;
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
	//Generate N random bodies with locations defined by bounds
	float4 *s = new float4[N];
	float3 *v = new float3[N];
	float3 *a = new float3[N];

	srand(time(NULL));
	for (int i = 0; i < N; i++) {
		s[i].x = ((float)rand() / RAND_MAX) * (upperX - lowerX);
		s[i].y = ((float)rand() / RAND_MAX) * (upperY - lowerY);
		s[i].z = ((float)rand() / RAND_MAX) * (upperZ - lowerZ);
		s[i].w = ((float)rand() / RAND_MAX) * (upperMass - lowerMass);

		//No initial velocity or acceleration 
		v[i].x = 0;
		v[i].y = 0;
		v[i].z = 0;
		a[i].x = 0;
		a[i].y = 0;
		a[i].z = 0;
	}

    return 0;
}