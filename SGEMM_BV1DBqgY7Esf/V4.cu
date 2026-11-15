#include<cstdio>
#define A(i,j) a[(i)*n+(j)]
#define B(i,j) b[(i)*n+(j)]
void random_matrix(int m,int n,float* a){
    int i,j;

    for(i=0;i<m;i++)
        for(j=0;j<n;j++)
#if 1    
            A(i,j)=2.0*(float)drand48()-1.0;
#else
            A(i,j)=(j-i)%3;
#endif
        }
float compare_matrix(int m,int n,float*a,float*b){
    int i,j;
    float max_diff=0.0,diff;
    int printed=0;
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            diff=abs(A(i,j)-B(i,j));
            max_diff=(diff>max_diff?diff:max_diff);
            if(0==printed)
                if(max_diff>0.5f||max_diff<-0.5f){
                    printf("\n error:i %d  j %d  diff %f  got %f expect %f",i,j,max_diff,A(i,j),B(i,j));
                    printed=1;
                }
        }
    }
    return max_diff;
}


void cpu_sgemm(float* A_ptr,float* B_ptr,float* C_ptr,int M,int N,int K){
    for(int m=0;m<M;m++){
        for(int n=0;n<N;n++){
            float tmp=0.f;
            for(int k=0;k<K;k++){
                tmp+=A_ptr[m*K+k]*B_ptr[k*N+n];
            }
            C_ptr[m*N+n]=tmp;
        }
    }

}


//用 float4
#define FETCH_FLOAT4(pointer)(reinterpret_cast<float4 *>(&(pointer))[0 ])
template<unsigned int M_NUM_PER_BLOCK,
         unsigned int N_NUM_PER_BLOCK,
         unsigned int K_NUM_PER_BLOCK,
         unsigned int NUM_PER_THREAD>
__global__ void cuda_sgemm(float* A_ptr,float* B_ptr,float* C_ptr,const int M,int K,int N){
    int  tx=threadIdx.x;
    int  ty=threadIdx.y;
    float *A_ptr_start=A_ptr+blockIdx.y*M_NUM_PER_BLOCK*K;
    float *B_ptr_start=B_ptr+blockIdx.x*N_NUM_PER_BLOCK;

    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    float tmp[NUM_PER_THREAD]={0.f}; 

    for(int s=0;s<K;s+=K_NUM_PER_BLOCK){
        FETCH_FLOAT4(a_shared[ty][tx*NUM_PER_THREAD]=A_ptr_start[ty*K+s+tx*NUM_PER_THREAD]);
        FETCH_FLOAT4(b_shared[ty][tx*NUM_PER_THREAD]=B_ptr_start[(ty+s)*N+tx*NUM_PER_THREAD]);
        __syncthreads();
    }

    for(int i=0;i<NUM_PER_THREAD;i++){
        for(int k=0;k<K_NUM_PER_BLOCK;k++)
            tmp[i]=a_shared[ty][k]*b_shared[k][tx*NUM_PER_THREAD+i];
    }
    __syncthreads();

    float *C_ptr_start=C_ptr+N*blockIdx.y*M_NUM_PER_BLOCK+blockIdx.x*N_NUM_PER_BLOCK;
    for(int i=0;i<NUM_PER_THREAD;i++){
        C_ptr_start[ty*N+tx*NUM_PER_THREAD+i]=tmp[i];
    }


}



int main(){
    int m=2048;
    int n=2048;
    int k=2028;
    const size_t mem_size_A=m*k*sizeof(float);
    const size_t mem_size_B=k*n*sizeof(float);
    const size_t mem_size_C=m*n*sizeof(float);

    float *matrix_A_host=(float*)malloc(mem_size_A);
    float *matrix_B_host=(float*)malloc(mem_size_B);

    float *matrix_C_cpu=(float*)malloc(mem_size_C);
    float *matrix_C_gpu=(float*)malloc(mem_size_C);

    random_matrix(m,k,matrix_A_host);
    random_matrix(k,n,matrix_B_host);
    memset(matrix_C_cpu,0,mem_size_C);
    memset(matrix_C_gpu,0,mem_size_C);

    float *matrix_A_device,*matrix_B_device,*matrix_C_device;
    cudaMalloc((void**)&matrix_A_device,mem_size_A);
    cudaMalloc((void**)&matrix_B_device,mem_size_B);
    cudaMalloc((void**)&matrix_C_device,mem_size_C);

    cudaMemcpy(matrix_A_device,matrix_A_host,mem_size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device,matrix_B_host,mem_size_B,cudaMemcpyHostToDevice);

    cpu_sgemm(matrix_A_host,matrix_B_host,matrix_C_cpu,m,n,k);

    constexpr int M_NUM_PER_BLOCK=32;
    constexpr int N_NUM_PER_BLOCK=32;
    constexpr int K_NUM_PER_BLOCK=32;
    constexpr int NUM_PER_THREAD=4;
    dim3 block(8,32);
    dim3 grid(m/M_NUM_PER_BLOCK,n/N_NUM_PER_BLOCK);
    
    cuda_sgemm<M_NUM_PER_BLOCK,N_NUM_PER_BLOCK,K_NUM_PER_BLOCK,NUM_PER_THREAD><<<grid,block>>>(matrix_A_device,matrix_B_device,matrix_C_device,m,n,k);

    cudaMemcpy(matrix_C_gpu,matrix_C_device,mem_size_C,cudaMemcpyDeviceToHost);
    
    float diff=compare_matrix(m,n,matrix_C_gpu,matrix_C_cpu);
    if(diff>0.5f||diff<-0.5f){
        printf("diff too big\n");
        exit(-1);
    }else{
        printf("right\n");
    }

    printf("sgemm\n");
    return 0;
}