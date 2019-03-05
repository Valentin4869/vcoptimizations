// keep old to compare.
// make new kernels for arbitrary sepconv
// possibly first/second kernels with different layouts to speedup maxpooling (first chw, after that hwc)


//This includes cl.h
#include <CL/cl.h>
#include <GeneralUtils.h>

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"

//include the generic inner product functions of ViennaCL
#include "viennacl/linalg/inner_prod.hpp"

//include the generic norm functions of ViennaCL
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/tools/random.hpp"

#define SELECTED_DEVICE 2




#define WIDTH 96
#define HEIGHT 96

//Fixed parameters
#define ROWS_BLOCKDIM_X  96
#define ROWS_BLOCKDIM_Y  4
#define ROWS_2_BLOCKDIM_X  48
#define ROWS_2_BLOCKDIM_Y  4

#define COLUMNS_BLOCKDIM_X  32
#define COLUMNS_BLOCKDIM_Y  8

#define COLUMNS_2_BLOCKDIM_X  16
#define COLUMNS_2_BLOCKDIM_Y  8

#define KERNEL_RADIUS 2
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1) 
#define C1 3
#define C2 7
#define C3 32
#define MP1_BLOCK_DIM 32
#define MP2_BLOCK_DIM 16
#define KERNELS_PATH "kernel.cl"
#define KERNEL_DIM 5

#define CL_ERROR(RET, OPNAME)	if (RET != CL_SUCCESS) {\
fprintf(stderr, "%s failed: %s\n" , OPNAME, uclErrorString(RET)); pause();  }


void pause() {

#ifdef _WIN32
	system("pause");
#elif __linux__
	//system("read -p \"Press any key to continue...\n\"");
#endif
}

void CompareResults(float* gold, float* cudaOutput, size_t n) {


	double sum = 0.0;
	for (int i = 0; i < n; i++)
		sum += (((double(gold[i]) - double(cudaOutput[i])))*((double(gold[i]) - double(cudaOutput[i]))));
	printf("L2: %f\n", sum);


	 sum = 0.0;
	for (int i = 0; i < n; i++)
		sum += (double)gold[i] - (double)cudaOutput[i];
	printf("L1: %f\n", sum);

	double sum_0 = 0.0;
	double sum_1 = 0.0;
	for (int i = 0; i < n; i++) {
		sum_0 += (double)cudaOutput[i];
		sum_1 += (double)gold[i];

	}
	printf("(L): %f\n", (sum_0 - sum_1));

	printf("Elementwise: ");
	for (int i = 0; i < n; i++)
		if (((float(gold[i]) - float(cudaOutput[i])))*((float(gold[i]) - float(cudaOutput[i]))) > 0.001)
		{
			printf("***DIFFERENCES FOUND***; first at %i (%f != %f).\n", i, gold[i], cudaOutput[i]);

			return;
		}

	printf("PASSED\n");

}



cl_device_id uclSetDevice(cl_int dev_id) {

	unsigned int i;
	int err;
	cl_uint num_platforms;
	cl_uint num_devices_total;
	cl_platform_id platform[8];
	cl_device_id devices[64];

	err |= clGetPlatformIDs(0, NULL, &num_platforms);

	err |= clGetPlatformIDs(num_platforms, platform, NULL);

	printf("Available platforms (%i):\n", num_platforms);

	for (i = 0; i < num_platforms; i++) {
		char name[64];
		char version[64];
		clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 64, name, NULL);
		clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, 64, version, NULL);
		printf(" %s %s\n", name, version);
	}

	num_devices_total = 0;


	for (i = 0; i < num_platforms; i++) {
		cl_uint num_devices;
		err |= clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		err |= clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, num_devices, &devices[num_devices_total], NULL);
		num_devices_total += num_devices;
	}

	printf("Available devices (%i):\n", num_devices_total);

	for (i = 0; i < num_devices_total; i++) {
		char name[64];
		cl_bool blImSupport;
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 64, name, NULL);
		clGetDeviceInfo(devices[i], CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &blImSupport, NULL);

		printf("[%i] %s. Image Support: %s\n", i, name, blImSupport ? "Yes" : "No");
	}
	printf("Using device %i\n", dev_id);


	return  devices[dev_id];





}

void rshw_NCHWtoHWCN(float *dst, float* src, int Dh, int Dw, int C, int F) {


	float* tmp = (float*)malloc(Dh*Dw*C*F * sizeof(float));


	for (int f = 0; f < F; f++)
		for (int c = 0; c < C; c++)
			for (int i = 0; i < Dh; i++)
				for (int j = 0; j < Dw; j++)
					tmp[i * Dw + j + c * Dw * Dh + f*C * Dh * Dw] = src[i * Dw * C*F + j*C*F + c*F + f ];


	for (int f = 0; f < F; f++)
		for (int c = 0; c < C; c++)
			for (int i = 0; i < Dh; i++)
				for (int j = 0; j < Dw; j++)
					src[i * Dw + j + c * Dw * Dh + f*C * Dw * Dh]=tmp[i * Dw + j + c * Dw * Dh + f*C * Dw * Dh] ;

	free(tmp);
}

void prfSepConv(int K1, int N1, int K2, int N2, double *prf) {

	cl_int	ret = 0;
	//profiling
	float s_t = 0.0f;
	cl_event start, stop;
	cl_ulong time_start, time_end;



	cl_context ctx = NULL;
	cl_command_queue cmdQ = NULL;
	cl_program	program = NULL;
	cl_device_id iDevice = uclSetDevice(SELECTED_DEVICE);
	ctx = clCreateContext(NULL, 1, &iDevice, NULL, NULL, &ret); CL_ERROR(ret, "clCreateContext");
	cmdQ = clCreateCommandQueue(ctx, iDevice, CL_QUEUE_PROFILING_ENABLE, &ret); CL_ERROR(ret, "clCreateCommandQueue");


	size_t szProgSource;
	size_t szBuildLog;
	char *cProgSource;
	char cBuildLog[100000];
	char cCompileOptions[2048 + 1];



#ifdef _WIN32
	sprintf_s(cCompileOptions, 2048, "\
                -D WIDTH=%u\
                -D HEIGHT=%u\
                -D KERNEL_RADIUS=%u\
                -D ROWS_BLOCKDIM_X=%u\
                -D ROWS_BLOCKDIM_Y=%u\
                -D ROWS_2_BLOCKDIM_X=%u\
                -D ROWS_2_BLOCKDIM_Y=%u\
                -D COLUMNS_BLOCKDIM_X=%u\
                -D COLUMNS_BLOCKDIM_Y=%u\
                -D COLUMNS_2_BLOCKDIM_X=%u\
                -D COLUMNS_2_BLOCKDIM_Y=%u\
                -D KERNEL_RADIUS=%u\
                -D KERNEL_LENGTH=%u\
                -D C1=%u\
                -D C2=%u\
                -D C3=%u\
                -D K1=%u\
                -D N1=%u\
                -D K2=%u\
                -D N2=%u\
                -D MP1_BLOCK_DIM=%u\
                -D MP2_BLOCK_DIM=%u",

		WIDTH,
		HEIGHT,
		KERNEL_RADIUS,
		ROWS_BLOCKDIM_X,
		ROWS_BLOCKDIM_Y,
		ROWS_2_BLOCKDIM_X,
		ROWS_2_BLOCKDIM_Y,
		COLUMNS_BLOCKDIM_X,
		COLUMNS_BLOCKDIM_Y,
		COLUMNS_2_BLOCKDIM_X,
		COLUMNS_2_BLOCKDIM_Y,
		KERNEL_RADIUS,
		KERNEL_LENGTH,
		C1,
		C2,
		C3,
		K1,
		N1,
		K2,
		N2,
		MP1_BLOCK_DIM,
		MP2_BLOCK_DIM);


#else //linux
	sprintf(cCompileOptions, "\
                -D WIDTH=%u\
                -D HEIGHT=%u\
                -D KERNEL_RADIUS=%u\
                -D ROWS_BLOCKDIM_X=%u\
                -D ROWS_BLOCKDIM_Y=%u\
                -D ROWS_2_BLOCKDIM_X=%u\
                -D ROWS_2_BLOCKDIM_Y=%u\
                -D COLUMNS_BLOCKDIM_X=%u\
                -D COLUMNS_BLOCKDIM_Y=%u\
                -D COLUMNS_2_BLOCKDIM_X=%u\
                -D COLUMNS_2_BLOCKDIM_Y=%u\
                -D KERNEL_RADIUS=%u\
                -D KERNEL_LENGTH=%u\
                -D C1=%u\
                -D C2=%u\
                -D C3=%u\
                -D MP1_BLOCK_DIM=%u\
                -D MP2_BLOCK_DIM=%u",

		WIDTH,
		HEIGHT,
		KERNEL_RADIUS,
		ROWS_BLOCKDIM_X,
		ROWS_BLOCKDIM_Y,
		ROWS_2_BLOCKDIM_X,
		ROWS_2_BLOCKDIM_Y,
		COLUMNS_BLOCKDIM_X,
		COLUMNS_BLOCKDIM_Y,
		COLUMNS_2_BLOCKDIM_X,
		COLUMNS_2_BLOCKDIM_Y,
		KERNEL_RADIUS,
		KERNEL_LENGTH,
		C1,
		C2,
		C3,
		MP1_BLOCK_DIM,
		MP2_BLOCK_DIM);


#endif

	cProgSource = uclLoadSource(KERNELS_PATH, &szProgSource);
	program = clCreateProgramWithSource(ctx, 1, (const char **)&cProgSource, (const size_t *)&szProgSource, &ret);
	CL_ERROR(ret, "CreateProgram");

	ret = (clBuildProgram(program, 1, &iDevice, cCompileOptions, NULL, NULL));


	if (ret != CL_SUCCESS)
	{
		clGetProgramBuildInfo(program, iDevice, CL_PROGRAM_BUILD_LOG, 10000 * sizeof(char), cBuildLog, &szBuildLog);
		clFinish(cmdQ);
		cBuildLog[szBuildLog - 1] = '\0';
		printf("*Build Log (%i): %s\n", szBuildLog, cBuildLog);
	}










	//Allocate host memory

	cl_float *v1 = (cl_float*)malloc(KERNEL_DIM * 1 * C1 * K1 * sizeof(cl_float));
	cl_float *h1 = (cl_float*)malloc(1 * KERNEL_DIM * K1 * N1 * sizeof(cl_float));
	cl_float *v2 = (cl_float*)malloc(KERNEL_DIM * 1 * N1 * K2 * sizeof(cl_float));
	cl_float *h2 = (cl_float*)malloc(1 * KERNEL_DIM * K2 * N2 * sizeof(cl_float));
	cl_float *input = (cl_float*)malloc(C1 * WIDTH * HEIGHT * sizeof(cl_float));
	cl_float *conv_out = (cl_float*)malloc(C3*WIDTH * HEIGHT * sizeof(cl_float));

	cl_float *out_v1 = (cl_float*)malloc(WIDTH*HEIGHT * K1 * sizeof(cl_float));
	cl_float *out_h1 = (cl_float*)malloc(WIDTH*HEIGHT * N1 * sizeof(cl_float));
	cl_float *out_v2 = (cl_float*)malloc(WIDTH / 2 * HEIGHT / 2 * K2 * sizeof(cl_float));
	cl_float *out_h2 = (cl_float*)malloc(WIDTH / 2 * HEIGHT / 2 * N2 * sizeof(cl_float));

	cl_float *gold_v1 = (cl_float*)malloc(WIDTH*HEIGHT * K1 * sizeof(cl_float));
	cl_float *gold_h1 = (cl_float*)malloc(WIDTH*HEIGHT * N1 * sizeof(cl_float));
	cl_float *gold_v2 = (cl_float*)malloc(WIDTH / 2 * HEIGHT / 2 * K2 * sizeof(cl_float));
	cl_float *gold_h2 = (cl_float*)malloc(WIDTH / 2 * HEIGHT / 2 * N2 * sizeof(cl_float));


	//Allocate device memory
	cl_mem dev_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY, C1 * WIDTH*HEIGHT * sizeof(cl_float), NULL, &ret);
	cl_mem dev_outV1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, K1 * WIDTH*HEIGHT * sizeof(cl_float), NULL, &ret);
	cl_mem dev_outH1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N1 * WIDTH*HEIGHT * sizeof(cl_float), NULL, &ret);
	cl_mem dev_outV2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, K2 * WIDTH / 2 * HEIGHT / 2 * sizeof(cl_float), NULL, &ret);
	cl_mem dev_outH2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N2 * WIDTH / 2 * HEIGHT / 2 * sizeof(cl_float), NULL, &ret);
	cl_mem dev_v1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 1 * KERNEL_DIM * C1 * K1 * sizeof(cl_float), NULL, &ret);
	cl_mem dev_h1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 1 * KERNEL_DIM * K1 * N1 * sizeof(cl_float), NULL, &ret);
	cl_mem dev_v2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 1 * KERNEL_DIM * N1 * K2 * sizeof(cl_float), NULL, &ret);
	cl_mem dev_h2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 1 * KERNEL_DIM * K2 * N2 * sizeof(cl_float), NULL, &ret);
	cl_mem dev_mxpH1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, WIDTH / 2 * HEIGHT / 2 * N1 * sizeof(cl_float), NULL, &ret);
	cl_mem dev_mxpH2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, WIDTH / 4 * HEIGHT / 4 * N2 * sizeof(cl_float), NULL, &ret);

	clEnqueueWriteBuffer(cmdQ, dev_input, CL_TRUE, 0, C1 * WIDTH*HEIGHT * sizeof(cl_float), input, 0, NULL, NULL); //input image buffer
	clEnqueueWriteBuffer(cmdQ, dev_v1, CL_TRUE, 0, 1 * KERNEL_DIM * C1 * K1 * sizeof(cl_float), v1, 0, NULL, NULL);
	clEnqueueWriteBuffer(cmdQ, dev_h1, CL_TRUE, 0, 1 * KERNEL_DIM * K1 * N1 * sizeof(cl_float), h1, 0, NULL, NULL);
	clEnqueueWriteBuffer(cmdQ, dev_v2, CL_TRUE, 0, 1 * KERNEL_DIM * N1 * K2 * sizeof(cl_float), v2, 0, NULL, NULL);
	clEnqueueWriteBuffer(cmdQ, dev_h2, CL_TRUE, 0, 1 * KERNEL_DIM * K2 * N2 * sizeof(cl_float), h2, 0, NULL, NULL);


	//kernels
	cl_kernel	krnRConv1 = NULL,
		krnCConv1 = NULL,
		krnRConv2 = NULL,
		krnCConv2 = NULL,
		krnMaxPool1 = NULL,
		krnMaxPool2 = NULL;



	printf("CreateKernel: \n");
	//create kernels
	krnRConv1 = clCreateKernel(program, "rowConv", &ret);
	printf("**rowConv: %s\n", uclErrorString(ret));

	krnCConv1 = clCreateKernel(program, "colConv", &ret);
	printf("**colConv: %s\n", uclErrorString(ret));

	krnMaxPool1 = clCreateKernel(program, "MaxPool1", &ret);
	printf("**MaxPool1: %s\n", uclErrorString(ret));

	krnRConv2 = clCreateKernel(program, "rowConv2", &ret);
	printf("**rowConv2: %s\n", uclErrorString(ret));

	krnCConv2 = clCreateKernel(program, "colConv2", &ret);
	printf("**colConv2: %s\n", uclErrorString(ret));

	krnMaxPool2 = clCreateKernel(program, "MaxPool2", &ret);
	printf("**MaxPool2: %s\n", uclErrorString(ret));


	clSetKernelArg(krnCConv1, 0, sizeof(cl_mem), (void*)&dev_input);
	clSetKernelArg(krnCConv1, 1, sizeof(cl_mem), (void*)&dev_outV1);
	clSetKernelArg(krnCConv1, 2, sizeof(cl_mem), (void*)&dev_v1);

	clSetKernelArg(krnRConv1, 0, sizeof(cl_mem), (void*)&dev_outV1); // in
	clSetKernelArg(krnRConv1, 1, sizeof(cl_mem), (void*)&dev_outH1); //out
	clSetKernelArg(krnRConv1, 2, sizeof(cl_mem), (void*)&dev_h1);

	clSetKernelArg(krnMaxPool1, 0, sizeof(cl_mem), (void*)&dev_outH1);
	clSetKernelArg(krnMaxPool1, 1, sizeof(cl_mem), (void*)&dev_mxpH1);


	clSetKernelArg(krnCConv2, 0, sizeof(cl_mem), (void*)&dev_mxpH1);
	clSetKernelArg(krnCConv2, 1, sizeof(cl_mem), (void*)&dev_outV2);
	clSetKernelArg(krnCConv2, 2, sizeof(cl_mem), (void*)&dev_v2);

	clSetKernelArg(krnRConv2, 0, sizeof(cl_mem), (void*)&dev_outV2); // in
	clSetKernelArg(krnRConv2, 1, sizeof(cl_mem), (void*)&dev_outH2); //out
	clSetKernelArg(krnRConv2, 2, sizeof(cl_mem), (void*)&dev_h2);

	clSetKernelArg(krnMaxPool2, 0, sizeof(cl_mem), (void*)&dev_outH2);
	clSetKernelArg(krnMaxPool2, 1, sizeof(cl_mem), (void*)&dev_mxpH2);



	//cols
	size_t global_cols1[] = { WIDTH , HEIGHT , K1 };
	size_t threads_cols1[] = { COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, 1 };
	//rows
	size_t global_rows1[] = { WIDTH , HEIGHT , N1 };
	size_t threads_rows1[] = { ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y, 1 };

	size_t global_MP1[] = { WIDTH, HEIGHT , N1 };
	size_t threads_MP1[] = { MP1_BLOCK_DIM, MP1_BLOCK_DIM, 1 };

	//cols2
	size_t global_cols2[] = { (WIDTH / 2) , (HEIGHT / 2), K2 };
	size_t threads_cols2[] = { COLUMNS_2_BLOCKDIM_X, COLUMNS_2_BLOCKDIM_Y, 1 };

	//rows2
	size_t global_rows2[] = { (WIDTH / 2), (HEIGHT / 2) , N2 };
	size_t threads_rows2[] = { ROWS_2_BLOCKDIM_X, ROWS_2_BLOCKDIM_Y, 1 };


	size_t global_MP2[] = { WIDTH / 2, HEIGHT / 2 , N2 };
	size_t threads_MP2[] = { MP2_BLOCK_DIM, MP2_BLOCK_DIM, 1 };


	printf("Warmup launch...\n");
	for (int i = 0; i < 2000; i++) {
		ret = clEnqueueNDRangeKernel(cmdQ, krnCConv1, 3, NULL, global_cols1, threads_cols1, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(cmdQ, krnRConv1, 3, NULL, global_rows1, threads_rows1, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(cmdQ, krnMaxPool1, 3, NULL, global_MP1, threads_MP1, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(cmdQ, krnCConv2, 3, NULL, global_cols2, threads_cols2, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(cmdQ, krnRConv2, 3, NULL, global_rows2, threads_rows2, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(cmdQ, krnMaxPool2, 3, NULL, global_MP2, threads_MP2, 0, NULL, NULL);
	}
	clFinish(cmdQ);


	printf("Launch Kernels...\n");


	for (int i = 0; i < 10000; i++) {
	ret = clEnqueueNDRangeKernel(cmdQ, krnCConv1, 3, NULL, global_cols1, threads_cols1, 0, NULL, &start);
	ret = clEnqueueNDRangeKernel(cmdQ, krnRConv1, 3, NULL, global_rows1, threads_rows1, 0, NULL, NULL);
	ret = clEnqueueNDRangeKernel(cmdQ, krnMaxPool1, 3, NULL, global_MP1, threads_MP1, 0, NULL, NULL);
	ret = clEnqueueNDRangeKernel(cmdQ, krnCConv2, 3, NULL, global_cols2, threads_cols2, 0, NULL, NULL);
	ret = clEnqueueNDRangeKernel(cmdQ, krnRConv2, 3, NULL, global_rows2, threads_rows2, 0, NULL, NULL);
	ret = clEnqueueNDRangeKernel(cmdQ, krnMaxPool2, 3, NULL, global_MP2, threads_MP2, 0, NULL, &stop);

	clWaitForEvents(1, &stop);
	//clFinish(cmdQ);
	clGetEventProfilingInfo(start, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(stop, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

	s_t += (time_end - time_start);
	}
	printf("Time elapsed: %fus\n\n\n", 1e+6*s_t*1e-9/10000.0);
	*prf = 1e+6*s_t*1e-9/10000.0;

	printf("\n\nCleanup OpenCL...\n");

	CL_ERROR(clFlush(cmdQ), "clFlush");
	CL_ERROR(clFinish(cmdQ), "clFinish");
	CL_ERROR(clReleaseKernel(krnRConv1), "clReleaseKernel");
	CL_ERROR(clReleaseKernel(krnCConv1), "clReleaseKernel");
	CL_ERROR(clReleaseKernel(krnRConv2), "clReleaseKernel");
	CL_ERROR(clReleaseKernel(krnCConv2), "clReleaseKernel");
	CL_ERROR(clReleaseKernel(krnMaxPool1), "clReleaseKernel");
	CL_ERROR(clReleaseKernel(krnMaxPool2), "clReleaseKernel");
	CL_ERROR(clReleaseProgram(program), "clReleaseProgram");

	CL_ERROR(clReleaseMemObject(dev_input), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_outV1), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_outH1), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_outV2), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_outH2), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_v1), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_h1), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_v2), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_h2), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_mxpH1), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_mxpH2), "clReleaseMemObject");

	CL_ERROR(clReleaseEvent(start), "clReleaseEvent");
	CL_ERROR(clReleaseEvent(stop), "clReleaseEvent");

	CL_ERROR(clReleaseDevice(iDevice), "clReleaseDevice");
	CL_ERROR(clReleaseCommandQueue(cmdQ), "clReleaseCommandQueue");
	CL_ERROR(clReleaseContext(ctx), "clReleaseContext");


	printf("Freeing host memory...\n");
	free(v1);
	free(h1);
	free(v2);
	free(h2);
	free(input);
	free(conv_out);




}

int main(int argc, char** argv)
{
	goto clpart;
	cl_int	ret = 0;
	//profiling
	float s_t = 0.0f;
	cl_event start, stop;
	cl_ulong time_start, time_end;



	cl_context ctx = NULL;
	cl_command_queue cmdQ = NULL;
	cl_program	program = NULL;
	cl_device_id iDevice = uclSetDevice(SELECTED_DEVICE);
	ctx = clCreateContext(NULL, 1, &iDevice, NULL, NULL, &ret); CL_ERROR(ret, "clCreateContext");
	cmdQ = clCreateCommandQueue(ctx, iDevice, CL_QUEUE_PROFILING_ENABLE, &ret); CL_ERROR(ret, "clCreateCommandQueue");


	size_t szProgSource;
	size_t szBuildLog;
	char *cProgSource;
	char cBuildLog[100000];
	char cCompileOptions[2048+1];


	int K1 = 7;
	int N1 = 32;
	int K2 = 7;
	int N2 = 32;



#ifdef _WIN32
	sprintf_s(cCompileOptions, 2048, "\
                -D WIDTH=%u\
                -D HEIGHT=%u\
                -D KERNEL_RADIUS=%u\
                -D ROWS_BLOCKDIM_X=%u\
                -D ROWS_BLOCKDIM_Y=%u\
                -D ROWS_2_BLOCKDIM_X=%u\
                -D ROWS_2_BLOCKDIM_Y=%u\
                -D COLUMNS_BLOCKDIM_X=%u\
                -D COLUMNS_BLOCKDIM_Y=%u\
                -D COLUMNS_2_BLOCKDIM_X=%u\
                -D COLUMNS_2_BLOCKDIM_Y=%u\
                -D KERNEL_RADIUS=%u\
                -D KERNEL_LENGTH=%u\
                -D C1=%u\
                -D C2=%u\
                -D C3=%u\
                -D K1=%u\
                -D N1=%u\
                -D K2=%u\
                -D N2=%u\
                -D MP1_BLOCK_DIM=%u\
                -D MP2_BLOCK_DIM=%u",

				WIDTH,
				HEIGHT,
				KERNEL_RADIUS,
				ROWS_BLOCKDIM_X,
				ROWS_BLOCKDIM_Y,
				ROWS_2_BLOCKDIM_X,
				ROWS_2_BLOCKDIM_Y,
				COLUMNS_BLOCKDIM_X,
				COLUMNS_BLOCKDIM_Y,
				COLUMNS_2_BLOCKDIM_X,
				COLUMNS_2_BLOCKDIM_Y,		
				KERNEL_RADIUS,
				KERNEL_LENGTH,
				C1,
				C2,
				C3,
				K1,
				N1,
				K2,
				N2,
				MP1_BLOCK_DIM,
				MP2_BLOCK_DIM);


#else //linux
	sprintf(cCompileOptions, "\
                -D WIDTH=%u\
                -D HEIGHT=%u\
                -D KERNEL_RADIUS=%u\
                -D ROWS_BLOCKDIM_X=%u\
                -D ROWS_BLOCKDIM_Y=%u\
                -D ROWS_2_BLOCKDIM_X=%u\
                -D ROWS_2_BLOCKDIM_Y=%u\
                -D COLUMNS_BLOCKDIM_X=%u\
                -D COLUMNS_BLOCKDIM_Y=%u\
                -D COLUMNS_2_BLOCKDIM_X=%u\
                -D COLUMNS_2_BLOCKDIM_Y=%u\
                -D KERNEL_RADIUS=%u\
                -D KERNEL_LENGTH=%u\
                -D C1=%u\
                -D C2=%u\
                -D C3=%u\
                -D MP1_BLOCK_DIM=%u\
                -D MP2_BLOCK_DIM=%u",

		WIDTH,
		HEIGHT,
		KERNEL_RADIUS,
		ROWS_BLOCKDIM_X,
		ROWS_BLOCKDIM_Y,
		ROWS_2_BLOCKDIM_X,
		ROWS_2_BLOCKDIM_Y,
		COLUMNS_BLOCKDIM_X,
		COLUMNS_BLOCKDIM_Y,
		COLUMNS_2_BLOCKDIM_X,
		COLUMNS_2_BLOCKDIM_Y,
		KERNEL_RADIUS,
		KERNEL_LENGTH,
		C1,
		C2,
		C3,
		MP1_BLOCK_DIM,
		MP2_BLOCK_DIM);


#endif

	cProgSource = uclLoadSource(KERNELS_PATH, &szProgSource);
	program = clCreateProgramWithSource(ctx, 1, (const char **)&cProgSource, (const size_t *)&szProgSource, &ret);
	CL_ERROR(ret, "CreateProgram");

	ret = (clBuildProgram(program, 1, &iDevice, cCompileOptions, NULL, NULL));


	if (ret != CL_SUCCESS)
	{
		clGetProgramBuildInfo(program, iDevice, CL_PROGRAM_BUILD_LOG, 10000 * sizeof(char), cBuildLog, &szBuildLog);
		clFinish(cmdQ);
		cBuildLog[szBuildLog - 1] = '\0';
		printf("*Build Log (%i): %s\n", szBuildLog, cBuildLog);
	}




	





	//Allocate host memory

	cl_float *v1 = (cl_float*)malloc(KERNEL_DIM * 1 * C1 * K1 * sizeof(cl_float));
	cl_float *h1 = (cl_float*)malloc(1 * KERNEL_DIM * K1 * N1 * sizeof(cl_float));
	cl_float *v2 = (cl_float*)malloc(KERNEL_DIM * 1 * N1 * K2 * sizeof(cl_float));
	cl_float *h2 = (cl_float*)malloc(1 * KERNEL_DIM * K2 * N2 * sizeof(cl_float));
	cl_float *input = (cl_float*)malloc(C1 * WIDTH * HEIGHT * sizeof(cl_float));
	cl_float *conv_out = (cl_float*)malloc(C3*WIDTH * HEIGHT * sizeof(cl_float));

	cl_float *out_v1 = (cl_float*)malloc(WIDTH*HEIGHT * K1 * sizeof(cl_float));
	cl_float *out_h1 = (cl_float*)malloc(WIDTH*HEIGHT * N1 * sizeof(cl_float));
	cl_float *out_v2 = (cl_float*)malloc(WIDTH / 2 * HEIGHT / 2 * K2 * sizeof(cl_float));
	cl_float *out_h2 = (cl_float*)malloc(WIDTH / 2 * HEIGHT / 2 * N2 * sizeof(cl_float));

	cl_float *gold_v1 = (cl_float*)malloc(WIDTH*HEIGHT * K1 * sizeof(cl_float));
	cl_float *gold_h1 = (cl_float*)malloc(WIDTH*HEIGHT * N1 * sizeof(cl_float));
	cl_float *gold_v2 = (cl_float*)malloc(WIDTH/2*HEIGHT/2 * K2 * sizeof(cl_float));
	cl_float *gold_h2 = (cl_float*)malloc(WIDTH/2*HEIGHT/2 * N2 * sizeof(cl_float));


	//uclLoadData("rc96.bin", input, C1 * WIDTH*HEIGHT);
	uclLoadData("C:\\Users\\Mir\\Documents\\Visual Studio 2015\\Projects\\vc1\\vc1\\scweights5\\gold\\x.csv", input, C1 * WIDTH*HEIGHT);

	uclLoadData("C:\\Users\\Mir\\Documents\\Visual Studio 2015\\Projects\\vc1\\vc1\\scweights5\\wV1.csv", v1, 1 * KERNEL_DIM * C1 * K1);
	uclLoadData("C:\\Users\\Mir\\Documents\\Visual Studio 2015\\Projects\\vc1\\vc1\\scweights5\\wH1.csv", h1, KERNEL_DIM * 1 * K1 * N1);
	uclLoadData("C:\\Users\\Mir\\Documents\\Visual Studio 2015\\Projects\\vc1\\vc1\\scweights5\\wV2.csv", v2, 1 * KERNEL_DIM * N1 * K2);
	uclLoadData("C:\\Users\\Mir\\Documents\\Visual Studio 2015\\Projects\\vc1\\vc1\\scweights5\\wH2.csv", h2, KERNEL_DIM * 1 * K2* N2);


	//uclDisplayMatrixf(v1, 1, 20,0);

	//rshw_NCHWtoHWCN(v1, v1, KERNEL_DIM,1, 3, C2);

	for (int i = 0; i <6; i++) {
		printf("wV1 (%i):\n", i);
		uclDisplayMatrixf(v1, 1, 5, 1 * 5 * i);
	}

	uclLoadData("C:\\Users\\Mir\\Documents\\Visual Studio 2015\\Projects\\vc1\\vc1\\scweights5\\gold\\v1.csv", gold_v1, WIDTH*HEIGHT * K1);
	uclLoadData("C:\\Users\\Mir\\Documents\\Visual Studio 2015\\Projects\\vc1\\vc1\\scweights5\\gold\\h1.csv", gold_h1, WIDTH*HEIGHT * N1);
	uclLoadData("C:\\Users\\Mir\\Documents\\Visual Studio 2015\\Projects\\vc1\\vc1\\scweights5\\gold\\v2.csv", gold_v2, WIDTH*HEIGHT/4 * K2);
	uclLoadData("C:\\Users\\Mir\\Documents\\Visual Studio 2015\\Projects\\vc1\\vc1\\scweights5\\gold\\h2.csv", gold_h2, WIDTH*HEIGHT/4 * N2);

	//Allocate device memory
	cl_mem dev_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY, C1 * WIDTH*HEIGHT * sizeof(cl_float), NULL, &ret); 
	cl_mem dev_outV1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, K1 * WIDTH*HEIGHT * sizeof(cl_float), NULL, &ret);
	cl_mem dev_outH1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE , N1 * WIDTH*HEIGHT * sizeof(cl_float), NULL, &ret);
	cl_mem dev_outV2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, K2 * WIDTH / 2 * HEIGHT / 2 * sizeof(cl_float), NULL, &ret);
	cl_mem dev_outH2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N2 * WIDTH / 2 * HEIGHT / 2 * sizeof(cl_float), NULL, &ret);
	cl_mem dev_v1 = clCreateBuffer(ctx,	CL_MEM_READ_ONLY, 1 * KERNEL_DIM * C1 * K1 * sizeof(cl_float), NULL, &ret);
	cl_mem dev_h1 = clCreateBuffer(ctx,	CL_MEM_READ_ONLY, 1 * KERNEL_DIM * K1 * N1 * sizeof(cl_float), NULL, &ret);
	cl_mem dev_v2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 1 * KERNEL_DIM * N1 * K2 * sizeof(cl_float), NULL, &ret);
	cl_mem dev_h2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 1 * KERNEL_DIM * K2 * N2 * sizeof(cl_float), NULL, &ret);
	cl_mem dev_mxpH1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, WIDTH/2*HEIGHT/2* N1 * sizeof(cl_float), NULL, &ret);
	cl_mem dev_mxpH2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, WIDTH / 4 * HEIGHT / 4 * N2 * sizeof(cl_float), NULL, &ret);

	clEnqueueWriteBuffer(cmdQ, dev_input, CL_TRUE, 0, C1 * WIDTH*HEIGHT * sizeof(cl_float), input, 0, NULL, NULL); //input image buffer
	clEnqueueWriteBuffer(cmdQ, dev_v1, CL_TRUE, 0, 1 * KERNEL_DIM * C1 * K1 * sizeof(cl_float), v1, 0, NULL, NULL);
	clEnqueueWriteBuffer(cmdQ, dev_h1, CL_TRUE, 0, 1 * KERNEL_DIM * K1 * N1 * sizeof(cl_float), h1, 0, NULL, NULL);
	clEnqueueWriteBuffer(cmdQ, dev_v2, CL_TRUE, 0, 1 * KERNEL_DIM * N1 * K2 * sizeof(cl_float), v2, 0, NULL, NULL);
	clEnqueueWriteBuffer(cmdQ, dev_h2, CL_TRUE, 0, 1 * KERNEL_DIM * K2 * N2 * sizeof(cl_float), h2, 0, NULL, NULL);


	printf("im:\n");
	uclDisplayMatrixf(input, 1, 20, 0);


	//kernels
	cl_kernel	krnRConv1 = NULL,
		krnCConv1 = NULL,
		krnRConv2 = NULL,
		krnCConv2 = NULL,
		krnMaxPool1 = NULL,
		krnMaxPool2 = NULL;



	printf("CreateKernel: \n");
	//create kernels
	krnRConv1 = clCreateKernel(program, "rowConv", &ret);
	printf("**rowConv: %s\n", uclErrorString(ret));

	krnCConv1 = clCreateKernel(program, "colConv", &ret);
	printf("**colConv: %s\n", uclErrorString(ret));

	krnMaxPool1 = clCreateKernel(program, "MaxPool1", &ret);
	printf("**MaxPool1: %s\n", uclErrorString(ret));

	krnRConv2 = clCreateKernel(program, "rowConv2", &ret);
	printf("**rowConv2: %s\n", uclErrorString(ret));

	krnCConv2 = clCreateKernel(program, "colConv2", &ret);
	printf("**colConv2: %s\n", uclErrorString(ret));

	krnMaxPool2 = clCreateKernel(program, "MaxPool2", &ret);
	printf("**MaxPool2: %s\n", uclErrorString(ret));


	clSetKernelArg(krnCConv1, 0, sizeof(cl_mem), (void*)&dev_input); 
	clSetKernelArg(krnCConv1, 1, sizeof(cl_mem), (void*)&dev_outV1); 
	clSetKernelArg(krnCConv1, 2, sizeof(cl_mem), (void*)&dev_v1);

	clSetKernelArg(krnRConv1, 0, sizeof(cl_mem), (void*)&dev_outV1); // in
	clSetKernelArg(krnRConv1, 1, sizeof(cl_mem), (void*)&dev_outH1); //out
	clSetKernelArg(krnRConv1, 2, sizeof(cl_mem), (void*)&dev_h1);

	clSetKernelArg(krnMaxPool1, 0, sizeof(cl_mem), (void*)&dev_outH1);
	clSetKernelArg(krnMaxPool1, 1, sizeof(cl_mem), (void*)&dev_mxpH1);


	clSetKernelArg(krnCConv2, 0, sizeof(cl_mem), (void*)&dev_mxpH1);
	clSetKernelArg(krnCConv2, 1, sizeof(cl_mem), (void*)&dev_outV2);
	clSetKernelArg(krnCConv2, 2, sizeof(cl_mem), (void*)&dev_v2);

	clSetKernelArg(krnRConv2, 0, sizeof(cl_mem), (void*)&dev_outV2); // in
	clSetKernelArg(krnRConv2, 1, sizeof(cl_mem), (void*)&dev_outH2); //out
	clSetKernelArg(krnRConv2, 2, sizeof(cl_mem), (void*)&dev_h2);

	clSetKernelArg(krnMaxPool2, 0, sizeof(cl_mem), (void*)&dev_outH2);
	clSetKernelArg(krnMaxPool2, 1, sizeof(cl_mem), (void*)&dev_mxpH2);



	//cols
	size_t global_cols1[]={WIDTH , HEIGHT , K1};
	size_t threads_cols1[]={COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, 1};
	//rows
	size_t global_rows1[]={WIDTH , HEIGHT , N1};
	size_t threads_rows1[]={ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y, 1};

	size_t global_MP1[]={WIDTH, HEIGHT , N1 };
	size_t threads_MP1[]={MP1_BLOCK_DIM, MP1_BLOCK_DIM, 1};

	//cols2
	size_t global_cols2[] = { (WIDTH / 2) , (HEIGHT / 2), K2 };
	size_t threads_cols2[] = { COLUMNS_2_BLOCKDIM_X, COLUMNS_2_BLOCKDIM_Y, 1 };

	//rows2
	size_t global_rows2[]={(WIDTH / 2), (HEIGHT / 2) , N2 };
	size_t threads_rows2[]={ROWS_2_BLOCKDIM_X, ROWS_2_BLOCKDIM_Y, 1};


	size_t global_MP2[]={WIDTH / 2, HEIGHT / 2 , N2 };
	size_t threads_MP2[]={MP2_BLOCK_DIM, MP2_BLOCK_DIM, 1};


	printf("Warmup launch...\n");
	for (int i = 0; i < 1000; i++) {
		ret = clEnqueueNDRangeKernel(cmdQ, krnCConv1, 3, NULL, global_cols1, threads_cols1, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(cmdQ, krnRConv1, 3, NULL, global_rows1, threads_rows1, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(cmdQ, krnMaxPool1, 3, NULL, global_MP1, threads_MP1, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(cmdQ, krnCConv2, 3, NULL, global_cols2, threads_cols2, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(cmdQ, krnRConv2, 3, NULL, global_rows2, threads_rows2, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(cmdQ, krnMaxPool2, 3, NULL, global_MP2, threads_MP2, 0, NULL, NULL);
	}
	clFinish(cmdQ);


	printf("Launch Kernels...\n");



		ret = clEnqueueNDRangeKernel(cmdQ, krnCConv1, 3, NULL, global_cols1, threads_cols1, 0, NULL, &start);
	//	clFinish(cmdQ);
	//	printf("**CConv: %s\n", uclErrorString(ret));

		ret = clEnqueueNDRangeKernel(cmdQ, krnRConv1, 3, NULL, global_rows1, threads_rows1, 0, NULL, NULL);
	//	clFinish(cmdQ);
	//	printf("**RConv: %s\n", uclErrorString(ret));

	
		ret = clEnqueueNDRangeKernel(cmdQ, krnMaxPool1, 3, NULL, global_MP1, threads_MP1, 0, NULL, NULL);
	//	clFinish(cmdQ);
	//	printf("*MaxPool1: %s\n", uclErrorString(ret));



		ret = clEnqueueNDRangeKernel(cmdQ, krnCConv2, 3, NULL, global_cols2, threads_cols2, 0, NULL, NULL);
	//	clFinish(cmdQ);
		//printf("**CConv2: %s\n", uclErrorString(ret));



		ret = clEnqueueNDRangeKernel(cmdQ, krnRConv2, 3, NULL, global_rows2, threads_rows2, 0, NULL, NULL);
	//	clFinish(cmdQ);

		
	//	printf("**RConv2: %s\n", uclErrorString(ret));


		ret = clEnqueueNDRangeKernel(cmdQ, krnMaxPool2, 3, NULL, global_MP2, threads_MP2, 0, NULL, &stop);
	//	clFinish(cmdQ);
	//	printf("*MaxPool2: %s\n", uclErrorString(ret));
		
		
		clWaitForEvents(1, &stop);
		//clFinish(cmdQ);
		clGetEventProfilingInfo(start, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(stop, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

		s_t += (time_end - time_start);

	printf("Time elapsed: %fus\n\n\n", 1e+6*s_t*1e-9);

	//---------------------------
	//	***Write outputs to file for verification in MATLAB*** //TODO: do verification here--- save matlab results in a gold folder
	//---------------------------
	ret=clEnqueueReadBuffer(cmdQ, dev_outV1, CL_TRUE, 0, K1 * WIDTH * HEIGHT * sizeof(cl_float), conv_out, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(cmdQ, dev_outV1, CL_TRUE, 0, K1 * WIDTH * HEIGHT * sizeof(cl_float), out_v1, 0, NULL, NULL);
	printf("*readv1: %s\n", uclErrorString(ret));

	uclWriteData("outV1.bin", conv_out, C2*WIDTH *HEIGHT);

	ret=clEnqueueReadBuffer(cmdQ, dev_outH1, CL_TRUE, 0, N1*WIDTH * HEIGHT * sizeof(cl_float), conv_out, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(cmdQ, dev_outH1, CL_TRUE, 0, N1*WIDTH * HEIGHT * sizeof(cl_float), out_h1, 0, NULL, NULL);
	printf("*readh1: %s\n", uclErrorString(ret));
	uclWriteData("outH1.bin", conv_out, N1*WIDTH *HEIGHT);

	ret = clEnqueueReadBuffer(cmdQ, dev_mxpH1, CL_TRUE, 0, N1 * WIDTH / 2 * HEIGHT/2 * sizeof(cl_float), conv_out, 0, NULL, NULL);
	
	printf("*readmxp1: %s\n", uclErrorString(ret));
	uclWriteData("mxpH1.bin", conv_out, N1 * WIDTH / 2 * HEIGHT / 2);

	ret = clEnqueueReadBuffer(cmdQ, dev_outV2, CL_TRUE, 0, K2 * WIDTH / 2 * HEIGHT / 2 * sizeof(cl_float), conv_out, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(cmdQ, dev_outV2, CL_TRUE, 0, K2 * WIDTH / 2 * HEIGHT / 2 * sizeof(cl_float), out_v2, 0, NULL, NULL);
	printf("*v2: %s\n", uclErrorString(ret));
	uclWriteData("outV2.bin", conv_out, K2 * WIDTH / 2 * HEIGHT / 2);

	ret = clEnqueueReadBuffer(cmdQ, dev_outH2, CL_TRUE, 0, N2 * WIDTH / 2 * HEIGHT / 2 * sizeof(cl_float), conv_out, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(cmdQ, dev_outH2, CL_TRUE, 0, N2 * WIDTH / 2 * HEIGHT / 2 * sizeof(cl_float), out_h2, 0, NULL, NULL);
	printf("*h2: %s\n", uclErrorString(ret));
	uclWriteData("outH2.bin", conv_out, N2 * WIDTH / 2 * HEIGHT / 2);

	ret = clEnqueueReadBuffer(cmdQ, dev_mxpH2, CL_TRUE, 0, N2 * WIDTH / 4 * HEIGHT / 4 * sizeof(cl_float), conv_out, 0, NULL, NULL);
	printf("*readmxp2: %s\n", uclErrorString(ret));
	uclWriteData("mxpH2.bin", conv_out, N2 * WIDTH / 4 * HEIGHT / 4);

	
	printf("Compare V1:");
	CompareResults(gold_v1, out_v1, K1 * WIDTH * HEIGHT);

	printf("\n\ngold_V1:");
	uclDisplayMatrix(gold_v1, 1, 20, 0);
	printf("\n V1:");
	uclDisplayMatrix(out_v1, 1, 20, 0);

	printf("\nCompare H1:");
	CompareResults(gold_h1, out_h1, N1 * WIDTH * HEIGHT);

	printf("\nCompare V2:");
	CompareResults(gold_v2, out_v2, K2 * WIDTH * HEIGHT/4);

	printf("\nCompare H2:");
	CompareResults(gold_h2, out_h2, N2 * WIDTH * HEIGHT/4);



	printf("\n\nCleanup OpenCL...\n");

	CL_ERROR(clFlush(cmdQ), "clFlush");
	CL_ERROR(clFinish(cmdQ), "clFinish");
	CL_ERROR(clReleaseKernel(krnRConv1), "clReleaseKernel");
	CL_ERROR(clReleaseKernel(krnCConv1), "clReleaseKernel");
	CL_ERROR(clReleaseKernel(krnRConv2), "clReleaseKernel");
	CL_ERROR(clReleaseKernel(krnCConv2), "clReleaseKernel");
	CL_ERROR(clReleaseKernel(krnMaxPool1), "clReleaseKernel");
	CL_ERROR(clReleaseKernel(krnMaxPool2), "clReleaseKernel");
	CL_ERROR(clReleaseProgram(program), "clReleaseProgram");

	CL_ERROR(clReleaseMemObject(dev_input), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_outV1), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_outH1), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_outV2), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_outH2), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_v1), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_h1), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_v2), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_h2), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_mxpH1), "clReleaseMemObject");
	CL_ERROR(clReleaseMemObject(dev_mxpH2), "clReleaseMemObject");

	CL_ERROR(clReleaseEvent(start), "clReleaseEvent");
	CL_ERROR(clReleaseEvent(stop), "clReleaseEvent");

	CL_ERROR(clReleaseDevice(iDevice), "clReleaseDevice");
	CL_ERROR(clReleaseCommandQueue(cmdQ), "clReleaseCommandQueue");
	CL_ERROR(clReleaseContext(ctx), "clReleaseContext");


	printf("Freeing host memory...\n");
	free(v1);
	free(h1);
	free(v2);
	free(h2);
	free(input);
	free(conv_out);



	pause();

clpart:
	double prfs[16];
	int i = 3;
	for (int i = 1; i <= 16; i++) {
		printf("K= %i\n",i);
		prfSepConv(i, 32, i, 32,&prfs[i-1]);
	}


	printf("[");
	for (int i = 1; i <= 16; i++) {
		printf(" %f%s", prfs[i - 1], i == 16 ? " ":",");
		//prfSepConv(i, 32, i, 32, &prfs[i - 1]);
	}printf("]\n");
	pause();
	return 0;
}