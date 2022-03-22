/* gpu_info.cu */
/* Get the information about GPU installed on the system */
/* SCV, December 2012 */

#include <stdio.h>

int main(int argc, char ** argv) {
   int deviceCount;
   cudaDeviceProp prop;

   /* Get the number of GPUs available */
   cudaGetDeviceCount(&deviceCount);

   /* For each GPU get the information */
   for (int i = 0; i < deviceCount; i++) {
    
      /* Get property of GPU #i */
      cudaGetDeviceProperties(&prop, i);

      /* Outpout information about GPU availability */
      if (i == 0) {
         if (prop.major == 9999 && prop.minor == 9999) {
            printf("No CUDA GPU has been detected\n");
            return -1;
         } else if (deviceCount == 1) {
            printf( "There is 1 device supporting CUDA\n");
         } else {
            printf( "There are %d  devices supporting CUDA\n", deviceCount);
         }
       }


      printf("\n\n ====== General Information for device %d ====== \n",i+1);
      printf(" Name: %s\n",prop.name );
      printf(" Compute capability: %d.%d\n", prop.major, prop.minor );
      printf(" Clock rate: %d\n",prop.clockRate );
	  
      printf(" Device copy overlap:  ");
      if(prop.deviceOverlap) printf(" Enabled\n");
      else printf(" Disabled\n");
	  
      printf(" Kernel execution timeout:  ");
      if(prop.kernelExecTimeoutEnabled)printf(" Enabled\n");
      else printf(" Disabled\n");
	  
      printf(" ----- Memory Information for device %d:\n",i+1);
      printf(" Total global memory: %ld (%d MB)\n", prop.totalGlobalMem, int(prop.totalGlobalMem*9.5367e-7));
      printf(" Total constant memory: %ld\n", prop.totalConstMem);
      printf(" Max memory pitch: %ld\n", prop.memPitch);
      printf(" Texture alignment: %ld\n", prop.textureAlignment);
	  
      printf(" ----- MP Information for device %d: \n",i+1);
      printf(" Multiprocessor count: %d\n", prop.multiProcessorCount);
      printf(" Shared mem per block: %ld\n", prop.sharedMemPerBlock);
      printf(" Registers per block: %d\n", prop.regsPerBlock);
      printf(" Threads in warp: %d\n", prop.warpSize);
      printf(" Max threads per block: %d\n", prop.maxThreadsPerBlock);
      printf(" Max thread dimensions: (%d %d %d)\n", prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
      printf(" Max grid dimensions: (%d %d %d)\n", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);
   }

   return 0;
}
