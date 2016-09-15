#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>
#include <sys/time.h>

#define NX 19				//the number of internal nodes in the x direction
#define NY 19				//the number of internal nodes in the y direction
#define SIZE (NX+2)*(NY+2)		//total number of nodes (including boundary nodes)

#define MIN_ITER 10000

__device__ double Tpre[SIZE];		//temperature of the previous timestep

__global__ void setup_kernel(curandState *state, double *dev_T)
{
	int i, j;
	int NThd_x;
	int idx;

	i = threadIdx.x + blockIdx.x*blockDim.x;
	j = threadIdx.y + blockIdx.y*blockDim.y;
	NThd_x = blockDim.x*gridDim.x;
	idx = i + j*NThd_x;

	curand_init(1234, idx, 0, &state[idx]);

	for(i=0; i<SIZE; i++)
		Tpre[i] = dev_T[i];
}



__global__ void kernel(double *T, double *para, curandState *state)
{
	double LX = para[0];		//length in x direction, m
	double LY = para[1];		//length in y direction, m
	double k = para[2];		//conductivity, W(m-K)
	double Q = para[3];		//internal heat source, W/m2
	double rho = para[4];		//density of the material, kg/m3
	double cp = para[5];		//specific heat capacity of the material, J/(kg-K)
	double dt = para[6];		//time step, sec

	int i;				//index in the x direction
	int j;				//index in the y direction
	int NThd_x;			//number of threads in the x direction
	int idx;			//index of the temperature nodes in 1D array	

	double r;			//random number
	double Tsum = 0.0;		//accumulated temperature
	double Told = 0.0;		//old value of the node's temperature
	double TH[] = {0.0, 0.0, 0.0,
		0.0};			//threshold value for random walking
	double q = 0.0;			//source term
	double dx = LX/(NX+1);		//spacial interval length in x direction, m
	double dy = LY/(NY+1);		//spacial interval length in y direction, m
	int pos_x = -1;			//current position of the node in the x direction
	int pos_y = -1;			//current position of the node in the y direction
	int flag = 1;			//flag for the outmost iteration
	int iflag = 0;			//flag for internal iteration
	int sflag = 0;			//flag for the stationary state
	int count = 0;			//number of iterations
	double err = 0.0;		//absolute error
	double const EPS = 1.0E-3;	//error tolerance

	double a[4];			//temporary storage the coefficients
	//
	NThd_x = blockDim.x*gridDim.x;
	j = threadIdx.x + blockIdx.x*blockDim.x;
	i = threadIdx.y + blockIdx.y*blockDim.y;

	idx = j + NThd_x*i;
	//save the result of the previous timestep
	Tpre[idx] = T[idx];
	//
	curandState localState = state[idx];
	//
	a[0] = rho*cp/dt;
	a[1] = k/(dx*dx);
	a[2] = k/(dy*dy);
	a[3] = a[0]+2.0*a[1]+2.0*a[2];

	TH[0] = a[1]/a[3];		//threshold for TW0
	TH[1] = TH[0]+a[1]/a[3];	//threshold for TE0
	TH[2] = TH[1]+a[2]/a[3];	//threshold for TS0
	TH[3] = TH[2]+a[2]/a[3];	//threshold for TN0
	q = Q/a[3];			//normalized source term 
	//
	if(i>0 && i<(NY+1) && j>0 && j<(NX+1))
	{
		while(flag)
		{	
			count++;
			Told = T[idx];
			Tsum = T[idx]*(count-1);
			pos_x = j;
			pos_y = i;
			iflag = 0;
			sflag = 0;
			//
			while(!iflag)
			{
				r = curand_uniform(&localState);
				//
				if(r<TH[0])
					//move to west
					pos_x--;
				else if(r<TH[1])
					//move to east
					pos_x++;
				else if(r<TH[2])
					//move to south
					pos_y--;
				else if(r<TH[3])
					//move to north
					pos_y++;
				else
				{
					pos_x += 0;
					pos_y += 0;
					sflag = 1;
				}
				//
				Tsum += q;
				if(sflag)
				{
					iflag = 1;
					Tsum += Tpre[pos_x+NThd_x*pos_y];
				}
				else if(pos_x == 0 || pos_x == NX+1 || pos_y == 0 || pos_y == NY+1)
				{
					iflag = 1;
					Tsum += T[pos_x+NThd_x*pos_y];
				}
			}
			//
			T[idx] = Tsum/count;
			err = fabs(T[idx]-Told);
			if(err<EPS && count>MIN_ITER)
			{
				flag = 0;
			}
			//state[idx] = localState;
		}
		//
		state[idx] = localState;
	}
}

int main()
{
	double const TW = 200.0;		//west boundary temperature, C
	double const TE = 150.0;		//east boundary temperature, C
	double const TS = 100.0;		//south boundary temperature, C
	double const TN = 50.0;			//north boundary temperature, C

	double const LX = 0.20;			//length in the x direction, m
	double const LY = 0.15;			//length in the y direction, m
	double const k = 385.0;			//conductivity, W(m-K)
	double const Q = 0.0;			//internal heat source, W/m2
	double const rho = 8.96E3;		//density, kg/m3
	double const cp = 3.85E2;		//specific capacity, J/(kg-K)

	int const NStep = 100;			//number of timestep
	double cal_time = 0.0;			//current time, sec
	double end_time = 10.0;			//end time
	double dt = end_time/NStep;		//time step, sec
	
	int const blocksize = 1;		//number of threads in each block
	dim3 dimBlock(blocksize, blocksize);
	dim3 dimGrid((NY+2+blocksize-1)/blocksize, (NX+2+blocksize-1)/blocksize);

	int i, j, t;
	double **Tfield;			//The field of temperature (2D)
	double *T;				//linearized temperature stored in 1D array
	double *dev_T;
	double Tinit = 0.25*(TW+TE+TS+TN);	//initial temperture, C
	double x;				//x-coordinate of node, m
	double y;				//y-coordinate of node, m
	double dx = LX/(NX+1);
	double dy = LY/(NY+1);
	
	double para[] 
		= {LX, LY, k, Q, 
		rho, cp, dt};			//parameters for solving
	double *dev_para;
	int num_para = sizeof(para)/sizeof(double);

	curandState *devStates;
	
	time_t startTime, endTime;
	struct timeval start, end;
	
	FILE *fp;				//file pointer
	//
	fp = fopen("Result.txt", "w");
	if(fp == NULL)
	{
		printf("Fail to open the result.txt file!");
	}
	//
	Tfield = (double**)malloc((NY+2)*sizeof(double*));
	for(i=0; i<NY+2; i++)
		Tfield[i] = (double*)malloc((NX+2)*sizeof(double));

	T = (double*)malloc(SIZE*sizeof(double));

	cudaMalloc((void**)&dev_T, SIZE*sizeof(double));
	cudaMalloc((void**)&devStates, SIZE*sizeof(curandState));
	cudaMalloc((void**)&dev_para, num_para*sizeof(double));
	//
	for(i=0; i<NY+2; i++)
	{
		for(j=0; j<NX+2; j++)
		{
			if(i == 0)
				Tfield[i][j] = TS;
			else if(i == NY+1)
				Tfield[i][j] = TN;
			else if(j == 0)
				Tfield[i][j] = TW;
			else if(j == NX+1)
				Tfield[i][j] = TE;
			else
				Tfield[i][j] = Tinit;
			//
			T[j+(NX+2)*i] = Tfield[i][j];
		}
	}
	//
	cudaMemcpy(dev_T, T, SIZE*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_para, para, num_para*sizeof(double), cudaMemcpyHostToDevice);
	//
	setup_kernel<<<dimGrid,dimBlock>>>(devStates, dev_T);
	//
	startTime = time(NULL);
	gettimeofday(&start, NULL);
	//
	for(t=0; t<NStep; t++)
	{
		cal_time += dt;
		//
		kernel<<<dimGrid, dimBlock>>>(dev_T, dev_para, devStates);
		//
		//cudaThreadSynchronize();
		cudaDeviceSynchronize();
		//	
		cudaMemcpy(T, dev_T, SIZE*sizeof(double), cudaMemcpyDeviceToHost);
		//
		fprintf(fp, "Results of time = %f:\n", cal_time);
		for(i=0; i<NY+2; i++)
		{
			for(j=0; j<NX+2; j++)
			{
				x = j*dx;
				y = i*dy;
				Tfield[i][j] = T[j+(NX+2)*i];
				fprintf(fp, "%f\t%f\t%f\n", x, y, Tfield[i][j]);
			}
		}
		fprintf(fp, "\n\n");
	}
	//
	endTime = time(NULL);
	gettimeofday(&end, NULL);
	printf("The calculation time is: %f seconds\n", difftime(endTime, startTime));
	double timelapse = (end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec)/1.0E6;
	printf("The time used for calculation is %f\n", timelapse);
	//
	cudaFree(dev_T);
	cudaFree(dev_para);
	fclose(fp);
	//
	return 0;
}

