#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)



//#define DEBUG

#ifdef  DEBUG
	#define print_debug(...) printf(__VA_ARGS__)
#else 
	#define print_debug(...) do {}while(0)
#endif //  

#define MIN(a,b) a>b ? b:a
#define CALC_Y(x,y,z) 0.299 * x + +0.587 * y + 0.114 * z

// текстурная ссылка <тип элементов, размерность, режим нормализации>
texture<uchar4, 2, cudaReadModeElementType> tex_RGB_format;


__device__ double mask_put(int x, int y, int* maska)
{
	double brightness = 0;
	double grad = 0;
	int index_l_array[3] = {-1,0,1};
	int size_mask_x = 3;
	int size_mask_y = 3;
	uchar4 u_format;
	// int maska_X[9] = { -1,0,1,-2,0,2,-1,0,1};
	
	for(int i=0;i<size_mask_y * size_mask_x;i++)
	{
		u_format = tex2D(tex_RGB_format, x +index_l_array[i%3], y - index_l_array[i/3]);
		brightness = 0.299 * u_format.x + 0.587 * u_format.y + 0.114 * u_format.z;//CALC_Y(u_format.x, u_format.y, u_format.z);
		grad += brightness*(double)maska[8 - i];

	}
	return grad;
}

// __device__ double mask_y(int x, int y)
// {
// 	double brightness, grad;
// 	int index_l_array[3] = {-1,0,1};
// 	const int size_mask_x = 3;
// 	const int size_mask_y = 3;
// 	uchar4 u_format;
// 	int maska_Y[9] = { -1, -2, -1, 0,0,0,1,2,1}; 
	
// 	for(int i=0;i<size_mask_y * size_mask_x;i++)
// 	{
// 		u_format = tex2D(tex_RGB_format, x +index_l_array[i%size_mask_x], y - index_l_array[i/size_mask_y]);
// 		brightness = 0.299*u_format.x+0.587*u_format.y + 0.114*u_format.z;//CALC_Y(u_format.x, u_format.y, u_format.z);
// 		grad+=brightness*(double)maska_Y[(size_mask_x*size_mask_y-1) - i];

// 	}
// 	return grad;
// }


__global__ void kernel(uchar4 *out, int height,int widht, int* maska_X, int* maska_Y)
{
	int indx = blockDim.x * blockIdx.x + threadIdx.x;
	int indy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	
	int x,y;

    uchar4 p_global;	

	// const int size_mask_x = 3;
	// const int size_mask_y = 3;
	

	
 	
	
	// double grad;
	// double temp_G;

	for (y = indy; y < height; y += offsety)
	{
		
		for (x = indx; x < widht; x += offsetx)
		{
            p_global = tex2D(tex_RGB_format, x,y);

			double gx = mask_put(x,y, maska_X);
			double gy = mask_put(x,y, maska_Y);
			
			double grad = sqrtf(gx*gx + gy*gy);
			// print_debug("gx = %d, gy = %d, grad = %d\n", gx,gy,grad);
			if(grad>UCHAR_MAX) { 
				// print_debug("max\n");
				grad = UCHAR_MAX;}
			out[y * widht + x] = make_uchar4(grad, grad, grad, p_global.w);
			// print_debug("out[%d] = %d\n",y * widht + x, out[y * widht + x]);


		}
	}
	
}

int main()
{

	
	int widht, height;
	char name_in[255];
    char name_out[255];
	scanf("%s",name_in);
    scanf("%s",name_out);

	FILE* file_read = fopen(name_in, "rb");
	
	if (NULL == file_read)
	{
		fprintf(stderr, "File read - error <do not found file>\n");
		return 0;
	}
	int size_mask = 9;
	int maska_X[9] = { -1,0,1,-2,0,2,-1,0,1};
	int maska_Y[9] = { -1, -2, -1, 0,0,0,1,2,1}; 
	
	
	fread(&widht, sizeof(int), 1, file_read);
	// print_debug("height = %d\n", widht);

	fread(&height, sizeof(int), 1, file_read);
	
	// print_debug("widht = %d\n", height);
	
	uchar4* image = (uchar4*)malloc(sizeof(uchar4) * widht* height);
	

	
	
	fread(image, sizeof(uchar4), widht * height, file_read);
	
		
	
	
	
	

	
	fclose(file_read);

	cudaArray *cuda_image;

	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();


	
	int *dev_maska_X;
	int *dev_maska_Y;

	CSC(cudaMalloc(&dev_maska_X, sizeof(int) *size_mask));
	CSC(cudaMalloc(&dev_maska_Y, sizeof(int) *size_mask));

	CSC(cudaMemcpy(dev_maska_X, maska_X, size_mask*sizeof(int),cudaMemcpyHostToDevice ));
	CSC(cudaMemcpy(dev_maska_Y, maska_Y, size_mask*sizeof(int),cudaMemcpyHostToDevice ));

	CSC(cudaMallocArray(&cuda_image, &ch, widht,height));

	CSC(cudaMemcpyToArray(cuda_image, 0, 0, image, sizeof(uchar4) * widht * height, cudaMemcpyHostToDevice));

	
	tex_RGB_format.addressMode[0] = cudaAddressModeClamp;	
	tex_RGB_format.addressMode[1] = cudaAddressModeClamp;
	tex_RGB_format.channelDesc    = ch;
	tex_RGB_format.filterMode     = cudaFilterModePoint;	
	tex_RGB_format.normalized     = false;

	CSC(cudaBindTextureToArray(tex_RGB_format, cuda_image, ch));


	uchar4* cuda_image_out;
	
	CSC(cudaMalloc(&cuda_image_out, sizeof(uchar4) * widht * height));

	kernel << < dim3(32, 32), dim3(32, 32) >> > (cuda_image_out,height,widht, dev_maska_X, dev_maska_Y);

	CSC(cudaGetLastError());

	CSC(cudaMemcpy(image, cuda_image_out, sizeof(uchar4) * widht * height, cudaMemcpyDeviceToHost));

	CSC(cudaUnbindTexture(tex_RGB_format));




	file_read = fopen(name_out, "wb");

	if (NULL == file_read)
	{
		fprintf(stderr, "File  write - error <do not found file>\n");
		return 0;
	}


	fwrite(&widht, sizeof(int), 1, file_read);
	fwrite(&height, sizeof(int), 1, file_read);
	fwrite(image, sizeof(uchar4), widht * height, file_read);
	fclose(file_read);


	CSC(cudaFreeArray(cuda_image));
	CSC(cudaFree(cuda_image_out));
	CSC(cudaFree(dev_maska_X));
	CSC(cudaFree(dev_maska_Y));
	free(image);
	return 0;
}
