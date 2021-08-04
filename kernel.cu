//////////////////////////////////////////////////
//          Project 01 PI Calculation           //
//////////////////////////////////////////////////
// C++
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
// CUDA C / C++
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda.h>

const double iterations = 250000000; /*MAX: 94906264*/

__host__ int printDevProp()
{
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	printf("==================================================\n");
	printf(" >>>>>>> PI Calculation with CPU and GPU <<<<<<<\n");
	printf(" - Device Name: %s\n", devProp.name);
	printf(" - Maximum number of threads per block: %d\n", devProp.maxThreadsPerBlock);
	printf(" - Number of iterations: %.1lf\n", iterations);
	printf("==================================================\n");
	return devProp.maxThreadsPerBlock;
}

__host__ void printStats(clock_t timer, dim3 dimGrid, dim3 dimBlock)
{
	printf(" - Total GPU time: %f ms.\n", ((((float)timer) / CLOCKS_PER_SEC) * 1000.0));
	printf(" - Total Threads: %d\n",
		dimGrid.x * dimGrid.y * dimGrid.z * dimBlock.x * dimBlock.y * dimBlock.z);
	printf(" - Configuracion de ejecucion: \n");
	printf("   + Grid [%d, %d, %d] Bloque [%d, %d, %d]\n",
		dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
}

__host__ double piCPU()
{
	double sum = 0;
	for (double i = 1; i < iterations; ++i)
	{
		sum += (1 / (i * i));
	}
	return sqrt(sum * 6);
}

__host__ int intDivision(double n, double m)
{
	int value = 0;
	if (((int)n % (int)m) == 0)
		value = (n / m);
	else
		value = (n / m) + 1;
	return value;
}

__global__ void crazyDivisions(double* ti, double iter)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ((tid) <= (int)iter - 1)
	{
		ti[tid] = (double)1 / (double)(((double)tid + (double)1) * ((double)tid + (double)1));
	}
}

__global__ void crazySums(double* ti01, double* ti02, double iter)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ((tid * 2) <= (int)iter)
	{
		ti02[tid] = ti01[(tid * 2)] + ti01[(tid * 2) + 1];
	}
}

__global__ void crazyPI(double* ti, double* pi)
{

	pi[0] = sqrt(ti[0] * 6);
}

int main(int argc, char* argv[])
{
	int maxThreads, numBlocks;
	double numpiCPU;
	double* totalItems01, * totalItems02, * numpiGPU;
	clock_t timer1, timer2;

	cudaFree(0);
	cudaSetDevice(0);

	maxThreads = printDevProp();

	printf("\t\t<<<<< CPU >>>>>\n");
	timer1 = clock();

	numpiCPU = piCPU();

	timer1 = clock() - timer1;

	printf(" - The value of PI in CPU is: %.8lf\n", numpiCPU); /*MAX DEC: 51*/
	printf(" - Total CPU time: %f ms.\n", ((((double)timer1) / CLOCKS_PER_SEC) * 1000.0));
	printf("==================================================\n");

	printf("\t\t<<<<< GPU >>>>>\n");

	cudaMalloc((void**)&totalItems01, iterations * sizeof(double));
	cudaMalloc((void**)&totalItems02, iterations * sizeof(double));
	cudaMalloc((void**)&numpiGPU, sizeof(double));

	timer2 = clock();

	numBlocks = intDivision(iterations, maxThreads);
	dim3 dimGrid(numBlocks);
	dim3 dimBlock(maxThreads);

	crazyDivisions << <dimGrid, dimBlock >> > (totalItems01, iterations);
	cudaDeviceSynchronize();

	dim3 dimGrid2(ceil((double)numBlocks / (double)2));
	bool flag = true;
	double iter = iterations;
	unsigned int blocks = 0;
	while (iter != 1)
	{
		if (flag)
		{
			crazySums << <dimGrid2, dimBlock >> > (totalItems01, totalItems02, iter);
			flag = false;
		}
		else
		{
			crazySums << <dimGrid2, dimBlock >> > (totalItems02, totalItems01, iter);
			flag = true;
		}
		cudaDeviceSynchronize();
		iter = ceil(iter / 2);
		numBlocks = ceil((double)numBlocks / (double)2);
		if (numBlocks == 0)
			dimGrid2 = { (unsigned int)numBlocks + 1 };
		else
			dimGrid2 = { (unsigned int)numBlocks };
	}

	if (flag)
		crazyPI << <1, 1 >> > (totalItems02, numpiGPU);
	else
		crazyPI << <1, 1 >> > (totalItems01, numpiGPU);
	cudaDeviceSynchronize();

	timer2 = clock() - timer2;

	double pipi = 0;
	cudaMemcpy(&pipi, numpiGPU, sizeof(double), cudaMemcpyDeviceToHost);

	printf(" - The value of PI in GPU is: %.8lf\n", pipi); /*MAX DEC: 51*/
	printStats(timer2, dimGrid, dimBlock);
	printf("==================================================\n");

	//free memory - GPU
	cudaFree(totalItems01);
	cudaFree(totalItems02);
	cudaFree(numpiGPU);

	system("pause");
	return 0;
}