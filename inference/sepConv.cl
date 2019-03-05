


//________________ Passed down predefs ________________
//WIDTH =96
//HEIGHT =96
//KERNEL_RADIUS= (wKrn-1)/2
//ROWS_BLOCKDIM_X = 32, COLUMNS_BLOCKDIM_X = 4,
//ROWS_BLOCKDIM_Y = 4, COLUMNS_BLOCKDIM_Y = 32,

#define pWIDTH WIDTH/2 //48
#define pHEIGHT HEIGHT/2 //48
#define p2WIDTH 24 
#define p2HEIGHT 24 
#define PAD 0
#define C1 3
#define C2 7
#define C3 32



#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1) //3

__kernel __attribute__((reqd_work_group_size(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y, 1)))
void rowConv(
	__global  float* d_Src,
	__global float *d_Dst,
	__constant float* c_Kernel) {

	__local float l_data[C2][ROWS_BLOCKDIM_Y][ROWS_BLOCKDIM_X + KERNEL_RADIUS * 2];
	__local float l_result[ROWS_BLOCKDIM_Y / 2][ROWS_BLOCKDIM_X / 2];
	const int lix = get_local_id(0);
	const int liy = get_local_id(1);
	const int gix = get_global_id(0);
	const int giy = get_global_id(1);
	const int giz = get_global_id(2);

	const int block_idx = (ROWS_BLOCKDIM_Y*get_group_id(1) + liy)*WIDTH + get_group_id(0)*ROWS_BLOCKDIM_X + lix; //src idx
	const int dst_idx = (ROWS_BLOCKDIM_Y*get_group_id(1) / 2 + liy)*WIDTH / 2 + get_group_id(0)*ROWS_BLOCKDIM_X / 2 + lix + giz*WIDTH*HEIGHT;



	//**There probably is a more efficient way to load the data into shared memory.
	for (int c = 0; c < C2; c++)
		l_data[c][liy][lix + 2] = d_Src[block_idx + c*WIDTH*HEIGHT];

	if (lix <2)
		for (int c = 0; c < C2; c++)
			l_data[c][liy][lix] = gix - KERNEL_RADIUS >= 0 ? d_Src[block_idx - KERNEL_RADIUS + c*WIDTH*HEIGHT] : 0;

	else if (lix>ROWS_BLOCKDIM_X - 3) {
		for (int c = 0; c < C2; c++)
			l_data[c][liy][lix + 4] = gix + KERNEL_RADIUS < WIDTH ? d_Src[block_idx + KERNEL_RADIUS + c*WIDTH*HEIGHT] : 0;
	}


	

	barrier(CLK_LOCAL_MEM_FENCE);


	float sum = 0;

	//Parallelization of this summation doesn't improve performance on a small GPU (GTX 750)
	for (int c = 0; c < C2; c++) {
		float C_sum = 0;

		for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {

			C_sum += c_Kernel[KERNEL_RADIUS - j + c*KERNEL_LENGTH + giz*KERNEL_LENGTH*C2] * l_data[c][liy][lix + j + KERNEL_RADIUS];

		}
		sum += C_sum;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	
	//ReLU and maxpooling stage:


	//put all odd-index values for comparison
	if (gix % 2)
		l_data[0][liy][lix] = sum > 0 ? sum : 0;


	

	barrier(CLK_LOCAL_MEM_FENCE);

	//compare all even-index values with next to it
	if (!(gix % 2)) {
		l_data[0][liy][lix] = sum > l_data[0][liy][lix + 1] ? sum : l_data[0][liy][lix + 1];
	}

	

	barrier(CLK_LOCAL_MEM_FENCE);

	//printf("%i ",get_local_id(0));
	//get max for all even y,x

	if (!((gix % 2) && (giy % 2)))
		l_result[liy / 2][lix / 2] = l_data[0][liy][lix] > l_data[0][liy + 1][lix] ? l_data[0][liy][lix] : l_data[0][liy + 1][lix];
	
	barrier(CLK_LOCAL_MEM_FENCE); 


//	printf("I'm %i...",lix);
	if (get_local_id(0) > (ROWS_BLOCKDIM_X / 2 - 1) || get_local_id(1) > (ROWS_BLOCKDIM_Y / 2 - 1))
		return;

		d_Dst[dst_idx] = l_result[liy][lix];



}



__kernel __attribute__((reqd_work_group_size(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, 1)))
void colConv(
	__global float *d_Src,
	__global float *d_Dst,
	__constant float *c_Kernel) {

	__local float l_data[C1][COLUMNS_BLOCKDIM_Y + KERNEL_RADIUS * 2][COLUMNS_BLOCKDIM_X];
	const int lix = get_local_id(0); //max: 9
	const int liy = get_local_id(1);
	const int giy = get_global_id(1);
	const int block_idx = (COLUMNS_BLOCKDIM_Y*get_group_id(1) + liy)*WIDTH + get_group_id(0)*COLUMNS_BLOCKDIM_X + lix; //global index 
	const int dst_idx = block_idx + get_global_id(2)*WIDTH*HEIGHT;

	
	for (int c = 0; c < C1; c++)
		l_data[c][liy + KERNEL_RADIUS][lix] = d_Src[block_idx + c*WIDTH * HEIGHT];

	if (liy < 2) 
		for (int c = 0; c < C1; c++)
			l_data[c][liy][lix] = giy - KERNEL_RADIUS >= 0 ? d_Src[block_idx - KERNEL_RADIUS*WIDTH + c*WIDTH * HEIGHT] : 0; //2 because blockdim y is 2 in this case; fill left side with zeros

	else if (liy > COLUMNS_BLOCKDIM_Y - 3) 
		for (int c = 0; c < C1; c++)
			l_data[c][liy + KERNEL_RADIUS * 2][lix] = get_global_id(1) + KERNEL_RADIUS < HEIGHT ? d_Src[block_idx + KERNEL_RADIUS*WIDTH + c*WIDTH * HEIGHT] : 0; //fill right side with zeros



	//sync

	barrier(CLK_LOCAL_MEM_FENCE);



	float sum = 0;
	float C_sum = 0;

#pragma unroll
	for (int c = 0; c < C1; c++) {
		C_sum = 0;


		for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {

			C_sum += c_Kernel[KERNEL_RADIUS - j + c*KERNEL_LENGTH + get_global_id(2)*KERNEL_LENGTH*C1] * l_data[c][liy + j + KERNEL_RADIUS ][lix];

		}
		sum += C_sum;
	}


	d_Dst[dst_idx] = sum;
	

}


__kernel __attribute__((reqd_work_group_size(ROWS_BLOCKDIM_X/2, ROWS_BLOCKDIM_Y, 1)))
void rowConv2(
	__global  float* d_Src,
	__global float *d_Dst,
	__constant float* c_Kernel) {

	__local float l_data[C2][ROWS_BLOCKDIM_Y][ROWS_BLOCKDIM_X/2 + KERNEL_RADIUS * 2];
	__local float l_result[ROWS_BLOCKDIM_Y/2][ROWS_BLOCKDIM_X/4];

	const int lix = get_local_id(0);
	const int liy = get_local_id(1);
	const int gix = get_global_id(0);


	const int block_idx = (ROWS_BLOCKDIM_Y*get_group_id(1) + liy)*pWIDTH + get_group_id(0)*ROWS_BLOCKDIM_X/2 + lix;
	const int dst_idx = pWIDTH*ROWS_BLOCKDIM_Y*get_group_id(1) / 4 + liy*pWIDTH / 2 + get_group_id(0)*ROWS_BLOCKDIM_X / 4 + lix + get_global_id(2)*pWIDTH*pHEIGHT;



	for (int c = 0; c < C2; c++)
		l_data[c][liy][lix + 2] = d_Src[block_idx + c*pWIDTH*pHEIGHT];

	if (lix <2)
		for (int c = 0; c < C2; c++)
			l_data[c][liy][lix] = gix - KERNEL_RADIUS >= 0 ? d_Src[block_idx - KERNEL_RADIUS + c*pWIDTH*pHEIGHT] : 0;

	else if (lix>ROWS_BLOCKDIM_X/2 - 3) {
		for (int c = 0; c < C2; c++)
			l_data[c][liy][lix + 4] = gix + KERNEL_RADIUS < pWIDTH ? d_Src[block_idx + KERNEL_RADIUS + c*pWIDTH*pHEIGHT] : 0;
	}



	barrier(CLK_LOCAL_MEM_FENCE);


	float sum = 0;
	float C_sum = 0;


	for (int c = 0; c < C2; c++) {
		C_sum = 0;

		for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {

			C_sum += c_Kernel[KERNEL_RADIUS - j + c*KERNEL_LENGTH + get_global_id(2)*KERNEL_LENGTH*C2] * l_data[c][liy][lix + j + KERNEL_RADIUS];

		}
		sum += C_sum;
	}

	
	//put all odd-index values for comparison
	if (get_global_id(0) % 2)
		l_data[0][liy][lix] = sum > 0 ? sum : 0;



	barrier(CLK_LOCAL_MEM_FENCE);

	//compare all even-index values with next to it
	if (!(get_global_id(0) % 2)) {
		l_data[0][liy][lix] = sum > l_data[0][liy][lix + 1] ? sum : l_data[0][liy][lix + 1];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	//get max for all even y,x
	if (!(get_global_id(1) % 2 && (get_global_id(0) % 2)))
		l_result[liy / 2][lix / 2] = l_data[0][liy][lix] > l_data[0][liy + 1][lix] ? l_data[0][liy][lix] : l_data[0][liy + 1][lix];


	//threads that don't have anything left to do leave now
	if (lix > (ROWS_BLOCKDIM_X / 4 - 1) || liy > (ROWS_BLOCKDIM_Y / 2 - 1))
		return;

	barrier(CLK_LOCAL_MEM_FENCE); //this is for the if statement before ^this one

	//	if (lix < ROWS_BLOCKDIM_X / 2  && liy < ROWS_BLOCKDIM_Y / 2 )
	d_Dst[dst_idx] = l_result[liy][lix];



}


//Each thread loads one pixel, except for edge threads: each edge thread loads 3 pixels (2 kernel_raidus, 1 itself)
//Each thread computes one pixel
__kernel __attribute__((reqd_work_group_size(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, 1)))
void colConv2(
	__global float *d_Src,
	__global float *d_Dst,
	__constant float *c_Kernel) {

	__local float l_data[C3][COLUMNS_BLOCKDIM_Y + KERNEL_RADIUS * 2][COLUMNS_BLOCKDIM_X];
	const int lix = get_local_id(0); //max: 9
	const int liy = get_local_id(1);
	const int giy = get_global_id(1);
	const int block_idx = (COLUMNS_BLOCKDIM_Y*get_group_id(1) + liy)*pWIDTH + get_group_id(0)*COLUMNS_BLOCKDIM_X + lix; //global index 
	const int dst_idx = block_idx+ get_global_id(2)*pWIDTH*pHEIGHT; //global index 


	for (int c = 0; c < C3; c++) 

		l_data[c][liy + KERNEL_RADIUS][lix] = d_Src[block_idx+ c*WIDTH * HEIGHT];
	
	if (liy < 2) 
		for (int c = 0; c < C3; c++) 
				l_data[c][liy][lix] = giy - KERNEL_RADIUS >= 0 ? d_Src[block_idx - KERNEL_RADIUS*pWIDTH + c*WIDTH * HEIGHT] : 0; //2 because blockdim y is 2 in this case; fill left side with zeros
	
	else if (liy > COLUMNS_BLOCKDIM_Y - 3) 
		for (int c = 0; c < C3; c++) 
				l_data[c][liy + KERNEL_RADIUS*2][lix] = giy + KERNEL_RADIUS < pHEIGHT ? d_Src[block_idx+KERNEL_RADIUS*pWIDTH + c*WIDTH * HEIGHT] : 0; //fill right side with zeros


	barrier(CLK_LOCAL_MEM_FENCE);


	float sum = 0;
	float C_sum = 0;


	for (int c = 0; c < C3; c++) {
		C_sum = 0;


		for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {

			C_sum += c_Kernel[KERNEL_RADIUS - j + c*KERNEL_LENGTH + get_global_id(2)*KERNEL_LENGTH*C3] * l_data[c][liy + j + KERNEL_RADIUS][lix];

		}
		sum += C_sum;
	}
	

	
	d_Dst[dst_idx] = sum;

}


