// LogoDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <cv.h>
#include <highgui.h>
using namespace cv;

#include <cuda.h>
#include <cuda_runtime.h>

#include "LogoDetection.h"

#include <iostream>
#include <fstream>

int _tmain(int argc, _TCHAR* argv[])
{
	Mat img_ref = imread("images/ReferenceLogo.jpg");

	Size tmpsize(0,0);
	//resize(img_ref,img_ref,tmpsize,0.2,0.2);
	resize(img_ref,img_ref,tmpsize,0.08,0.08);
	//resize(img_ref,img_ref,tmpsize,0.1,0.1);

	Mat img_gray; 
	cvtColor(img_ref,img_gray,CV_RGB2GRAY);

	Mat img_gray32f;
	img_gray.convertTo(img_gray32f,CV_32F);

	//Mat img_tar = imread("images/curved1_0.12.png");
	Mat img_tar = imread("images/mixture1_008.jpg");
	//Mat img_tar = imread("images/flat1_020.gif");
	//Mat img_tar = imread("images/cloth4.jpg");
	//Mat img_tar = imread("images/planar1.jpg");
	//resize(img_tar,img_tar,tmpsize,4,4);

	Mat img_tar_gray;
	cvtColor(img_tar,img_tar_gray,CV_RGB2GRAY);

	Mat img_tar_gray32f;
	img_tar_gray.convertTo(img_tar_gray32f,CV_32F);

	Size imsize;
	imsize.height = img_gray32f.rows;
	imsize.width = img_gray32f.cols;

	float* imageArr;
	cudaMalloc((void **)&imageArr,imsize.width*imsize.height*sizeof(float));
	cudaMemcpy(imageArr, img_gray32f.data, imsize.width*imsize.height * sizeof(float), cudaMemcpyHostToDevice);

	Size imsizetar;
	imsizetar.height = img_tar_gray32f.rows;
	imsizetar.width = img_tar_gray32f.cols;

	float* imageTar;
	cudaMalloc((void **)&imageTar,imsizetar.width*imsizetar.height*sizeof(float));
	cudaMemcpy(imageTar, img_tar_gray32f.data, imsizetar.width*imsizetar.height * sizeof(float), cudaMemcpyHostToDevice);

	float *ret = computeHOG(imageArr,imageTar,imsize.width,imsize.height,imsizetar.width,imsizetar.height);

	int num = int(ret[0]);

	for(int i=1;i<=num;i++)
	{
		int logoX = int(ret[i])%imsizetar.width;
		int logoY = int(ret[i])/imsizetar.width;
		Point pt1(logoX-imsize.width/2,logoY-imsize.height/2);
		Point pt2(logoX+imsize.width/2,logoY+imsize.height/2);
		rectangle(img_tar,pt1,pt2,Scalar(0,0,255),4);
	}

	imshow("image", img_tar);
	imwrite("result2.jpg",img_tar);


	/*Mat voteImg(imsizetar.height,imsizetar.width,CV_32F);
	voteImg.data = (unsigned char*)(ret);

	double minvalue;
	double maxvalue;
	Point minindex;
	Point maxindex;

	minMaxLoc(voteImg,&minvalue,&maxvalue,&minindex,&maxindex);


	Mat tmp;
	voteImg.convertTo(tmp,CV_8U,255.0/maxvalue);

	imshow("image", tmp);
	imwrite("result2_vote.jpg",tmp);*/

	waitKey();
	return 0;
}

	/*std::ofstream myfile;
	myfile.open("example.txt");
	for(int i=0;i<8836;i++)//390 81 //65 97
	{
		myfile<<gradientX[i];
		myfile<<"\n";
	}
	myfile.close();*/

	/*std::ofstream myfile;
	myfile.open("example.txt");
	for(int i=0;i<180;i++)//390 81 //65 97
	{
		for(int j=0;j<181;j++)
		{
			//printf("%f\t",gradientX[i*81+j]);
			//Scalar intensity = img_gray.at<uchar>(i,j);
			//myfile<<intensity.val[0]<<"\t";
			myfile<<gradientX[i*181+j]<<"\t";
		}
		//printf("\n");
		myfile<<"\n";
	}
	myfile.close();*/