#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#include <stdlib.h>
#include <vector>
#include <algorithm>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>

#define BINNUM 9
#define CELLSIZE 6
#define CELLBLOCKSIZE 3
#define STEPSIZE 3
#define K 5
#define KERNEL_WIDTH 31

__global__ void computeGradient(float* outputGr,int* outputTag,float* input,unsigned int width,unsigned int height)
{
	unsigned int tx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y*blockDim.y + threadIdx.y; 

	float gradientX, gradientY;
	float angle;
	int nAngle;
	int indTag;

	if(tx < width && ty < height)
	{
		if(tx == 0) //image boundary
			gradientX = input[tx+1 + ty*width] - input[tx + ty*width];
		else if(tx == width-1)
			gradientX = input[tx + ty*width] - input[tx-1 + ty*width];
		else if(ty == 0 || ty == height-1)
			gradientX = 0;
		else
			gradientX = input[tx+1 + ty*width] - input[tx-1 + ty*width];

		
		if(ty == 0) //image boundary
			gradientY = input[tx + (ty+1)*width] - input[tx + (ty)*width];
		else if (ty == height-1)
			gradientY = input[tx + (ty)*width] - input[tx + (ty-1)*width];
		else if (tx ==0 || tx == width-1)
			gradientY = 0;
		else
			gradientY = input[tx + (ty+1)*width] - input[tx + (ty-1)*width];

		outputGr[tx + ty*width] = sqrt(gradientX*gradientX + gradientY*gradientY);
		//outputGr[tx + ty*width] = gradientX;

		if(gradientX == 0)
			gradientX = 1e-5;
		//angle = ((atan(gradientY/gradientX)));
		angle = ((atan(gradientY/gradientX)+(CUDART_PI_F/2))*180)/CUDART_PI_F;
		nAngle = 180/BINNUM;
		indTag = ceil(angle/nAngle);
		if(indTag == 0)
			indTag=1;
		else if(indTag==10)
			indTag=9;


		outputTag[tx + ty*width] = indTag;
	}

	
	//outputX[tx + ty*width] = gradientX;
	//outputY[tx + ty*width] = gradientY;
}

__global__ void computeBinHOG(float* outputHOGFeature,float* outputWeight,int *outputPosition,int* outputOffset,float* Gr,int* Tag,unsigned int xStepNum,unsigned int yStepNum,
								unsigned int width,unsigned int height)
{
	unsigned int tx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y*blockDim.y + threadIdx.y; 

	int centerX = floor(float(width)/2.0+0.5) - 1;
	int centerY = floor(float(height)/2.0+0.5) - 1;

	if(tx < xStepNum && ty < yStepNum)
	{
	//pixel position of center
	int x = tx*STEPSIZE+1.5*CELLSIZE-1;
	int y = ty*STEPSIZE+1.5*CELLSIZE-1;

	int leftupperX = x - 1.5*CELLSIZE +1 ;
	int leftupperY = y - 1.5*CELLSIZE +1 ;

	//__shared__ float tmp[CELLSIZE][CELLSIZE];
	float tmp[CELLBLOCKSIZE*CELLBLOCKSIZE][BINNUM] = {0.0f};

	int inc = 0;
	for(int i = 0; i < CELLBLOCKSIZE;i++)
	{
		for(int j=0; j<CELLBLOCKSIZE;j++)
		{
			int indX = leftupperX + j*CELLSIZE;
			int indY = leftupperY + i*CELLSIZE;
			for(int p=0;p<CELLSIZE;p++)
			{
				for(int q=0;q<CELLSIZE;q++)
				{
					//int currendindX = indX+p;
					//int currendindY = indY+q;
					int binind = Tag[indX+q + width*(indY+p)]-1;
					float tmpdebug = Gr[indX+q + width*(indY+p)];
					tmp[inc][binind] +=tmpdebug;
					//tmp[inc][binind] += Gr[indX+q + width*(indY+p)];
				}
			}
			inc++;
		}
	}

	float norm = 0.0f;
	int nonempty = 0;
	for(int p=0;p<CELLBLOCKSIZE*CELLBLOCKSIZE;p++)
	{
		for(int q=0;q<BINNUM;q++)
		{
			norm+= (tmp[p][q]*tmp[p][q]);
			if(tmp[p][q]>0)
				nonempty ++;
		}
	}
	norm = sqrt(norm);
	norm +=1e-5;
	//if(norm <= 0.0f)
	//	norm = 1e-5;
	//float acc = 0.0;
	for(int p=0;p<CELLBLOCKSIZE*CELLBLOCKSIZE;p++)
	{
		for(int q=0;q<BINNUM;q++)
		{
			//outputHOGFeature[ (tx+ty*xStepNum)*CELLBLOCKSIZE*CELLBLOCKSIZE*BINNUM +(q+p*BINNUM)] = tmp[p][q]/norm;
			outputHOGFeature[ (ty+tx*yStepNum)*CELLBLOCKSIZE*CELLBLOCKSIZE*BINNUM +(q+p*BINNUM)] = tmp[p][q]/norm;
			//acc +=tmp[p][q]/norm;
		}
	}
	/*outputPosition[(tx + ty*xStepNum)*2] = x;
	outputPosition[(tx + ty*xStepNum)*2 + 1] = y;

	outputOffset[(tx + ty*xStepNum)*2] = centerX - x;
	outputOffset[(tx + ty*xStepNum)*2 + 1] = centerY - y;

	outputWeight[tx + ty*xStepNum] = float(nonempty)/81.0f;*/

	outputPosition[(ty+tx*yStepNum)*2] = x;
	outputPosition[(ty+tx*yStepNum)*2 + 1] = y;

	outputOffset[(ty+tx*yStepNum)*2] = centerX - x;
	outputOffset[(ty+tx*yStepNum)*2 + 1] = centerY - y;

	outputWeight[(ty+tx*yStepNum)] = float(nonempty)/81.0f;
	//outputWeight[(ty+tx*yStepNum)] = float(acc)/1.0f;
	}
}

__global__ void computeBinHOGTar(float* outputHOGFeature,int *outputPosition,float* Gr,int* Tag,unsigned int xStepNum,unsigned int yStepNum,
								unsigned int width,unsigned int height)
{
	unsigned int tx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y*blockDim.y + threadIdx.y; 

	if(tx < xStepNum && ty < yStepNum)
	{
	//pixel position of center
	int x = tx*STEPSIZE+1.5*CELLSIZE-1;
	int y = ty*STEPSIZE+1.5*CELLSIZE-1;

	int leftupperX = x - 1.5*CELLSIZE +1 ;
	int leftupperY = y - 1.5*CELLSIZE +1 ;

	//__shared__ float tmp[CELLSIZE][CELLSIZE];
	float tmp[CELLBLOCKSIZE*CELLBLOCKSIZE][BINNUM] = {0.0f};

	int inc = 0;
	for(int i = 0; i < CELLBLOCKSIZE;i++)
	{
		for(int j=0; j<CELLBLOCKSIZE;j++)
		{
			int indX = leftupperX + j*CELLSIZE;
			int indY = leftupperY + i*CELLSIZE;
			for(int p=0;p<CELLSIZE;p++)
			{
				for(int q=0;q<CELLSIZE;q++)
				{
					//int currendindX = indX+p;
					//int currendindY = indY+q;
					int binind = Tag[indX+q + width*(indY+p)]-1;
					tmp[inc][binind] += Gr[indX+q + width*(indY+p)];
				}
			}
			inc++;
		}
	}

	float norm = 0.0f;
	int nonempty = 0;
	for(int p=0;p<CELLBLOCKSIZE*CELLBLOCKSIZE;p++)
	{
		for(int q=0;q<BINNUM;q++)
		{
			norm+= (tmp[p][q]*tmp[p][q]);
			if(tmp[p][q]>0)
				nonempty ++;
		}
	}
	norm = sqrt(norm);
	norm +=1e-5;
	//if(norm <= 0.0f)
	//	norm = 1e-5;
	for(int p=0;p<CELLBLOCKSIZE*CELLBLOCKSIZE;p++)
	{
		for(int q=0;q<BINNUM;q++)
		{
			//outputHOGFeature[ (tx+ty*xStepNum)*CELLBLOCKSIZE*CELLBLOCKSIZE*BINNUM +(q+p*BINNUM)] = tmp[p][q]/norm;
			outputHOGFeature[ (ty+tx*yStepNum)*CELLBLOCKSIZE*CELLBLOCKSIZE*BINNUM +(q+p*BINNUM)] = tmp[p][q]/norm;
		}
	}
	//outputPosition[(tx + ty*xStepNum)*2] = x;
	//outputPosition[(tx + ty*xStepNum)*2 + 1] = y;
	outputPosition[(ty+tx*yStepNum)*2] = x;
	outputPosition[(ty+tx*yStepNum)*2 + 1] = y;
	}
}

__global__ void computeDistance(float* output,float* hog,float* hogTar,unsigned int width,unsigned int height)
{
	/*__shared__ float tmp[2][81];
	unsigned int bx = blockIdx.x * 1;
	unsigned int by = blockIdx.y * 1;
	unsigned int tx = threadIdx.x * 1;

	if(tx < 81)
	{
		tmp[0][tx] = hog[tx + bx*CELLBLOCKSIZE*CELLBLOCKSIZE*BINNUM];
		tmp[1][tx] = hogTar[tx + by*CELLBLOCKSIZE*CELLBLOCKSIZE*BINNUM];
	}
		__syncthreads();

		float perDistance=0.0;
		float tmp1 = 0.0;
		float tmp2 = 0.0;
		for(int i=0;i<81;i++)
		{
			tmp1 = tmp[0][i];
			tmp2 = tmp[1][i];
			perDistance += ((tmp1-tmp2)*(tmp1-tmp2))/(tmp1 + tmp2 + 1e-5); 
		}

		output[bx + by*width] = 0.5 * perDistance;*/

	unsigned int tx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y*blockDim.y + threadIdx.y; 

	if(tx < width && ty < height)
	{
		float perDistance = 0.0;
		//float tmp[CELLBLOCKSIZE*CELLBLOCKSIZE*BINNUM]= {0.0f};
		//float tmpTar[CELLBLOCKSIZE*CELLBLOCKSIZE*BINNUM]= {0.0f};
		float tmp = 0.0;
		float tmpTar = 0.0;
		//int i=0;
		for(int i =0;i<CELLBLOCKSIZE*CELLBLOCKSIZE*BINNUM;i++)
		{

			tmp = hog[i + tx*CELLBLOCKSIZE*CELLBLOCKSIZE*BINNUM];
			tmpTar = hogTar[i + ty*CELLBLOCKSIZE*CELLBLOCKSIZE*BINNUM];

			//float tmp = hog[i*(width) + tx];
			//float tmpTar = hogTar[i*height + ty];

			perDistance = perDistance + ((tmp-tmpTar)*(tmp-tmpTar))/(tmp + tmpTar + 1e-5);
			//perDistance=1;
		}

		output[tx + ty*width] = 0.5 * perDistance;
		//output[ty + tx*height] = 0.5 * perDistance;
	}
}

__global__ void vote(float *OutputVoteMatrix,float *HOGDistance, float *weight,int *offset,int *samplepixel,
	                   unsigned int width,unsigned int height,unsigned int voteWidth,unsigned int voteHeight)
{
	unsigned int tx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y*blockDim.y + threadIdx.y; 
	if(tx < width && ty < height)
	{
		//if (HOGDistance[tx + ty*width] <= threshold[ty])
		if (HOGDistance[tx + ty*width] <= 2)//
		{
			int x = samplepixel[ty*2];
			int y = samplepixel[ty*2+1];

			int offsetx = offset[tx*2];
			int offsety = offset[tx*2+1];

			int centerx = x + offsetx;
			int centery = y + offsety;

			if(centerx >= 0 && centery >= 0 && centerx < voteWidth && centery< voteHeight)
			{
				OutputVoteMatrix[centerx + centery*voteWidth] += 1/(HOGDistance[tx + ty*width]+1e-5 + 1) * weight[tx];
			}
		}
	}
}

__global__ void genGaussianFilter(float* output)
{
	unsigned int tx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y*blockDim.y + threadIdx.y;

	__shared__ float gaussianKernel[KERNEL_WIDTH][KERNEL_WIDTH];

	float sigma = 5.0;
	float gaussianSum = 0.0;

	gaussianKernel[ty][tx] = powf(2.71828,-((tx-15)*(tx-15)/(2*sigma*sigma) + (ty-15)*(ty-15)/(2*sigma*sigma)));
	__syncthreads();

	for(int i=0;i<KERNEL_WIDTH;i++)
		for(int j=0;j<KERNEL_WIDTH;j++)
		{
			gaussianSum += gaussianKernel[i][j];
		}

	output[ty*KERNEL_WIDTH + tx] = gaussianKernel[ty][tx]/gaussianSum;
}

__global__ void gaussianFilter(float* output,float* input, float* gaussianKernel,unsigned int width,unsigned int height)
{
	unsigned int tx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y*blockDim.y + threadIdx.y;

	if(tx < width && ty < height)
	{
		float accum = 0.0;
		float offset = 15;
		for (int i = 0; i < KERNEL_WIDTH; ++i)
		{
			for (int j = 0; j < KERNEL_WIDTH; ++j)
			{
				int ind = tx+j-offset + (ty+i-offset) * width;
				if(ind > 0 && ind < width*height)
					accum += gaussianKernel[j + KERNEL_WIDTH*i] * input[ind];
			}

			output[tx + ty*width] = accum;
		}	
	}
}

__global__ void detectMaximal(float* output,float* input,unsigned int width,unsigned int height,float maximal)
{
	unsigned int tx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y*blockDim.y + threadIdx.y;
	if(tx < width-1 && ty < height-1 && tx >0 && ty>0)
	{
		if (input[tx + ty*width] > 0.8*maximal && input[tx + ty*width] > input[tx-1 + (ty-1)*width] && 
			input[tx + ty*width] > input[tx + (ty-1)*width] && input[tx + ty*width] > input[tx+1 + (ty-1)*width] &&
			input[tx + ty*width] > input[tx-1 + (ty)*width] && input[tx + ty*width] > input[tx+1 + (ty)*width] &&
			input[tx + ty*width] > input[tx-1 + (ty+1)*width] && input[tx + ty*width] > input[tx + (ty+1)*width] &&
			input[tx + ty*width] > input[tx+1 + (ty+1)*width])
		{
			output[tx + ty*width] = (tx+1) + (ty)*width;
		}
	}
}

__global__ void computeThreshold(float* output,float* input)
{
	unsigned int tx = threadIdx.x;
	unsigned int bx = blockIdx.x;
	__shared__ float partialMin[500];
	partialMin[tx] = input[tx + bx*blockDim.x];

	unsigned int stride = 1;

	for (stride = 1; stride < blockDim.x; stride *= 2) 
	{
	  __syncthreads();
	  if (tx % (2 * stride) == 0 && tx + stride < blockDim.x)
		  partialMin[tx] = partialMin[tx] < partialMin[tx + stride] ? partialMin[tx]:partialMin[tx + stride];
	}
	/*unsigned int stridenew = stride/2;
	float kmax = partialMin[0];

	for(int i=0;i<blockDim.x;i+=stridenew)
	{
		if(kmax < partialMin[i])
			kmax = partialMin[i];
	}*/

	output[0] = partialMin[0];
}

extern "C" float* computeHOG(float *img,float* imgTar,unsigned int width,unsigned int height,unsigned int widthTar,unsigned int heightTar)
{	
	//For performance analysis
	cudaEvent_t start, stop0,stop1, stop2, stop3, stop4, stop5, stop6,stop7,stop8,stop9,stop10,stop11,stop12,stop13,stop14;
	cudaEventCreate(&start);
	cudaEventCreate(&stop0);
	cudaEventCreate(&stop1);
	cudaEventCreate(&stop2);
	cudaEventCreate(&stop3);
	cudaEventCreate(&stop4);
	cudaEventCreate(&stop5);
	cudaEventCreate(&stop6);
	cudaEventCreate(&stop7);
	cudaEventCreate(&stop8);
	cudaEventCreate(&stop9);
	cudaEventCreate(&stop10);
	cudaEventCreate(&stop11);
	cudaEventCreate(&stop12);
	cudaEventCreate(&stop13);
	cudaEventCreate(&stop14);

	cudaEventRecord(start, 0);

	int xStepNum = floor((width-CELLSIZE*CELLBLOCKSIZE)/float(STEPSIZE));
	int yStepNum = floor((height-CELLSIZE*CELLBLOCKSIZE)/float(STEPSIZE));
	int xStepNumTar = floor((widthTar-CELLSIZE*CELLBLOCKSIZE)/float(STEPSIZE));
	int yStepNumTar = floor((heightTar-CELLSIZE*CELLBLOCKSIZE)/float(STEPSIZE));

	float *Gr;
	cudaMalloc((void**)&Gr , width*height*sizeof(float));

	int *Tag;
	cudaMalloc((void**)&Tag , width*height*sizeof(int));

	float *HOGFeature;
	cudaMalloc((void**)&HOGFeature , xStepNum*yStepNum*BINNUM*CELLBLOCKSIZE*CELLBLOCKSIZE*sizeof(float));
	cudaMemset(HOGFeature,0,xStepNum*yStepNum*BINNUM*CELLBLOCKSIZE*CELLBLOCKSIZE*sizeof(float));

	int *centerPosition;
	cudaMalloc((void**)&centerPosition , 2*xStepNum*yStepNum*sizeof(int));

	int *offset;
	cudaMalloc((void**)&offset , 2*xStepNum*yStepNum*sizeof(int));

	float *weight;
	cudaMalloc((void**)&weight , xStepNum*yStepNum*sizeof(float));

	
	float *GrTar;
	cudaMalloc((void**)&GrTar , widthTar*heightTar*sizeof(float));

	int *TagTar;
	cudaMalloc((void**)&TagTar , widthTar*heightTar*sizeof(int));

	int *samplepixel;
	cudaMalloc((void**)&samplepixel , 2*xStepNum*yStepNum*sizeof(int));

	float *HOGFeatureTar;
	cudaMalloc((void**)&HOGFeatureTar , xStepNumTar*yStepNumTar*BINNUM*CELLBLOCKSIZE*CELLBLOCKSIZE*sizeof(float));
	cudaMemset(HOGFeatureTar,0,xStepNumTar*yStepNumTar*BINNUM*CELLBLOCKSIZE*CELLBLOCKSIZE*sizeof(float));

	float *HOGDistance;
	cudaMalloc((void**)&HOGDistance , xStepNumTar*yStepNumTar*xStepNum*yStepNum*sizeof(float));
	cudaMemset(HOGDistance,0,xStepNumTar*yStepNumTar*xStepNum*yStepNum*sizeof(float));

	float *votematrix;
	cudaMalloc((void**)&votematrix , widthTar*heightTar* sizeof(float));
	cudaMemset(votematrix,0,widthTar*heightTar*sizeof(float));

	float *gaussianKernel_dev;
	cudaMalloc((void**)&gaussianKernel_dev , KERNEL_WIDTH*KERNEL_WIDTH* sizeof(float));
	cudaMemset(gaussianKernel_dev,0,KERNEL_WIDTH*KERNEL_WIDTH*sizeof(float));

	float *votematrix_smoothed;
	cudaMalloc((void**)&votematrix_smoothed , widthTar*heightTar* sizeof(float));
	cudaMemset(votematrix_smoothed,0,widthTar*heightTar*sizeof(float));

	float *vote_maxposition;
	cudaMalloc((void**)&vote_maxposition , widthTar*heightTar* sizeof(float));
	cudaMemset(vote_maxposition,0,widthTar*heightTar*sizeof(float));

	//********************
	cudaEventRecord(stop0, 0);
	cudaEventSynchronize(stop0);
	float elapsedTime0;
	cudaEventElapsedTime(&elapsedTime0, start, stop0);
	printf("Initialization :%f ms\n", elapsedTime0);
	//********************

	dim3 dimBlock(16,16,1);
	dim3 dimGrid(width/dimBlock.x + 1, height/dimBlock.y + 1,1);
	computeGradient<<<dimGrid,dimBlock>>>(Gr,Tag,img,width,height);

	//********************
	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	float elapsedTime1;
	cudaEventElapsedTime(&elapsedTime1, start, stop1);
	printf("Gradient Computation for Ref Image :%f ms\n", elapsedTime1);
	//********************

	dimBlock = dim3(16,16,1);
	dimGrid = dim3(xStepNum/dimBlock.x + 1, yStepNum/dimBlock.y + 1,1);
	computeBinHOG<<<dimGrid,dimBlock>>>(HOGFeature,weight,centerPosition,offset,Gr,Tag,xStepNum,yStepNum,width,height);
	
	//********************
	cudaEventRecord(stop2, 0);
	cudaEventSynchronize(stop2);
	float elapsedTime2;
	cudaEventElapsedTime(&elapsedTime2, start, stop2);
	printf("HOG Computation for Ref Image :%f ms\n", elapsedTime2);
	//********************

	dimBlock = dim3(16,16,1);
	dimGrid = dim3(widthTar/dimBlock.x + 1, heightTar/dimBlock.y + 1,1);
	computeGradient<<<dimGrid,dimBlock>>>(GrTar,TagTar,imgTar,widthTar,heightTar);

	//********************
	cudaEventRecord(stop3, 0);
	cudaEventSynchronize(stop3);
	float elapsedTime3;
	cudaEventElapsedTime(&elapsedTime3, start, stop3);
	printf("Gradient Computation for Target Image :%f ms\n", elapsedTime3);
	//********************

	dimBlock = dim3(16,16,1);
	dimGrid = dim3(xStepNumTar/dimBlock.x + 1, yStepNumTar/dimBlock.y + 1,1);
	computeBinHOGTar<<<dimGrid,dimBlock>>>(HOGFeatureTar,samplepixel,GrTar,TagTar,xStepNumTar,yStepNumTar,widthTar,heightTar);

	//********************
	cudaEventRecord(stop4, 0);
	cudaEventSynchronize(stop4);
	float elapsedTime4;
	cudaEventElapsedTime(&elapsedTime4, start, stop4);
	printf("HOG Computation for Target Image :%f ms\n", elapsedTime4);
	//********************

	dimBlock = dim3(16,16,1);
	dimGrid = dim3((xStepNum*yStepNum)/dimBlock.x + 1, (xStepNumTar*yStepNumTar)/dimBlock.y + 1,1);
	computeDistance<<<dimGrid,dimBlock>>>(HOGDistance,HOGFeature,HOGFeatureTar,xStepNum*yStepNum,xStepNumTar*yStepNumTar);

	//********************
	cudaEventRecord(stop5, 0);
	cudaEventSynchronize(stop5);
	float elapsedTime5;
	cudaEventElapsedTime(&elapsedTime5, start, stop5);
	printf("Distance Matrix Computation:%f ms\n", elapsedTime5);
	//********************

	//float *HOGDistanceSorted = new float[xStepNumTar*yStepNumTar*xStepNum*yStepNum];
	//cudaMemcpy(HOGDistanceSorted,HOGDistance,  xStepNumTar*yStepNumTar*xStepNum*yStepNum*sizeof(float), cudaMemcpyDeviceToHost);

	/*float *threshold_host  = new float[xStepNumTar*yStepNumTar];

	for(int i=0;i<xStepNumTar*yStepNumTar;i++)
	{
		thrust::sort(HOGDistanceSorted + i*(xStepNum*yStepNum), HOGDistanceSorted +i*(xStepNum*yStepNum) +xStepNum*yStepNum);
		threshold_host[i] = HOGDistanceSorted[i*(xStepNum*yStepNum) + K - 1];
	}*/

	/*float *threshold_dev;
	cudaMalloc((void**)&threshold_dev , xStepNumTar*yStepNumTar* sizeof(float));
	cudaMemcpy(threshold_dev,threshold_host, xStepNumTar*yStepNumTar*sizeof(float), cudaMemcpyHostToDevice);*/
	//cudaMemcpy(threshold_dev,threshold_host, xStepNumTar*yStepNumTar*sizeof(float), cudaMemcpyHostToDevice);

	/*dimBlock = dim3(xStepNum*yStepNum,1,1);
	dimGrid = dim3(xStepNumTar*yStepNumTar,1,1);
	float *threshold_dev;
	cudaMalloc((void**)&threshold_dev , xStepNumTar*yStepNumTar* sizeof(float));
	cudaMemset(threshold_dev,0,xStepNumTar*yStepNumTar*sizeof(float));
	computeThreshold<<<dimGrid,dimBlock>>>(threshold_dev,HOGDistance);*/



	dimBlock = dim3(16,16,1);
	dimGrid = dim3((xStepNum*yStepNum)/dimBlock.x + 1, (xStepNumTar*yStepNumTar)/dimBlock.y + 1,1);
	//vote<<<dimGrid,dimBlock>>>(votematrix,threshold_dev,HOGDistance,weight,offset,samplepixel,xStepNum*yStepNum,xStepNumTar*yStepNumTar,widthTar,heightTar);
	vote<<<dimGrid,dimBlock>>>(votematrix,HOGDistance,weight,offset,samplepixel,xStepNum*yStepNum,xStepNumTar*yStepNumTar,widthTar,heightTar);

	//********************
	cudaEventRecord(stop6, 0);
	cudaEventSynchronize(stop6);
	float elapsedTime6;
	cudaEventElapsedTime(&elapsedTime6, start, stop6);
	printf("Threshold and Voting:%f ms\n", elapsedTime6);
	//********************

	dimBlock = dim3(KERNEL_WIDTH,KERNEL_WIDTH,1);
	dimGrid = dim3(1,1,1);
	genGaussianFilter<<<dimGrid,dimBlock>>>(gaussianKernel_dev);

	//********************
	cudaEventRecord(stop7, 0);
	cudaEventSynchronize(stop7);
	float elapsedTime7;
	cudaEventElapsedTime(&elapsedTime7, start, stop7);
	printf("Gaussian Kernel Generation:%f ms\n", elapsedTime7);
	//********************

	dimBlock = dim3(16,16,1);
	dimGrid = dim3(widthTar/dimBlock.x + 1, heightTar/dimBlock.y + 1,1);
	gaussianFilter<<<dimGrid,dimBlock>>>(votematrix_smoothed,votematrix, gaussianKernel_dev,widthTar,heightTar);

	//********************
	cudaEventRecord(stop8, 0);
	cudaEventSynchronize(stop8);
	float elapsedTime8;
	cudaEventElapsedTime(&elapsedTime8, start, stop8);
	printf("Gaussian Blur:%f ms\n", elapsedTime8);
	//********************

	float *votematrix_smoothed_sorted = new float[widthTar*heightTar];
	cudaMemcpy(votematrix_smoothed_sorted,votematrix_smoothed,  widthTar*heightTar*sizeof(float), cudaMemcpyDeviceToHost);

	thrust::sort(votematrix_smoothed_sorted, votematrix_smoothed_sorted +widthTar*heightTar);
	float maxvalue = votematrix_smoothed_sorted[widthTar*heightTar-1];

	//********************
	cudaEventRecord(stop9, 0);
	cudaEventSynchronize(stop9);
	float elapsedTime9;
	cudaEventElapsedTime(&elapsedTime9, start, stop9);
	printf("Maximal Voting Value Computation:%f ms\n", elapsedTime9);
	//********************

	dimBlock = dim3(16,16,1);
	dimGrid = dim3(widthTar/dimBlock.x + 1, heightTar/dimBlock.y + 1,1);
	detectMaximal<<<dimGrid,dimBlock>>>(vote_maxposition,votematrix_smoothed,widthTar,heightTar,maxvalue);

	//********************
	cudaEventRecord(stop10, 0);
	cudaEventSynchronize(stop10);
	float elapsedTime10;
	cudaEventElapsedTime(&elapsedTime10, start, stop10);
	printf("Local Maximal Detection:%f ms\n", elapsedTime10);
	//********************

	float *vote_maxposition_sorted = new float[widthTar*heightTar];
	cudaMemcpy(vote_maxposition_sorted,vote_maxposition,  widthTar*heightTar*sizeof(float), cudaMemcpyDeviceToHost);

	thrust::sort(vote_maxposition_sorted, vote_maxposition_sorted +widthTar*heightTar);

	int centernum = 0;
	for(int i=widthTar*heightTar-1;i>=0;i--)
	{
		if(vote_maxposition_sorted[i] > 0)
			centernum++;
		else
			break;
	}

	//********************
	cudaEventRecord(stop11, 0);
	cudaEventSynchronize(stop11);
	float elapsedTime11;
	cudaEventElapsedTime(&elapsedTime11, start, stop11);
	printf("Transfering Detected Logo Position to HOST:%f ms\n", elapsedTime11);
	//********************

	//float *ret = new float[centernum];
	//cudaMemcpy(ret,vote_maxposition_sorted + widthTar*heightTar - centernum,  centernum*sizeof(float), cudaMemcpyHostToHost);

	float *ret = new float[centernum+1];
	ret[0] = float(centernum);
	cudaMemcpy(ret+1,vote_maxposition_sorted + widthTar*heightTar - centernum,  centernum*sizeof(float), cudaMemcpyHostToHost);

	//float *ret = new float[widthTar*heightTar];
	//cudaMemcpy(ret,votematrix_smoothed,  widthTar*heightTar*sizeof(float), cudaMemcpyDeviceToHost);

	return ret;
}
