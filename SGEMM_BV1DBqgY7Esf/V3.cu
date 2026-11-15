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


//增加每个线程工作
template<unsigned int BLOCK_SIZE,unsigned int STRIDE>
__global__ void cuda_sgemm(float* A_ptr,float* B_ptr,float* C_ptr,const int M,int K,int N){
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    constexpr int STEP=STRIDE*BLOCK_SIZE;

    float* A_ptr_start=A_ptr+STEP*blockIdx.y*K;
    float* B_ptr_start=B_ptr+STEP*blockIdx.x;
    float tmp[STRIDE][STRIDE]={0.f};

    __shared__ float a_shared[STEP][STEP];
    __shared__ float b_shared[STEP][STEP];



    for(int s=0;s<K;s+=STEP){
        //a_shared[threadIdx.y][threadIdx.x]=A_ptr_start[threadIdx.y*K+threadIdx.x+s];
        //b_shared[threadIdx.y][threadIdx.x]=B_ptr_start[(threadIdx.y+s)*N+threadIdx.x];
        for(int i=;i<STRIDE;i++){
            for(int j=0;j<STRIDE;j++){
                a_shared[ty+i*BLOCK_SIZE][tx+j*BLOCK_SIZE]=A_ptr_start[(ty+i*BLOCK_SIZE)*K+BLOCK_SIZE*j+tx+s];
                b_shared[ty+i*BLOCK_SIZE][tx+j*BLOCK_SIZE]=B_ptr_start[(ty+i*BLOCK_SIZE+s)*N+BLOCK_SIZE*j+tx];
            }
        }
        __syncthreads();
    }

    
    for(int i=0;i<STRIDE;i++){
            for(int j=;j<STRIDE;j++){
                for(int k=0;k<STEP;k++){
                     tmp[i][j]+=a_shared[ty+i*BLOCK_SIZE][k]*b_shared[k][tx+j*BLOCK_SIZE];
            }
        }
        __syncthreads();
    }


    float *C_ptr_start=C_ptr+N*blockIdx.y*STEP+blockIdx.x*STEP;
    for(int i=0;i<STRIDE;i++){
        for(int j=0;j<STRIDE;j++){
            C_ptr_start[(ty+i*BLOCK_SIZE)*N+tx+j*BLOCK_SIZE]=tmp[i][j];
        }
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

    constexpr int BLOCK=16;
    constexpr int STRIDE=2;
    dim3 block(BLOCK,BLOCK);
    dim3 grid((m+BLOCK-1)/BLOCK/STRIDE,(n+BLOCK-1)/BLOCK/STRIDE);
    
    cuda_sgemm<BLOCK,STRIDE><<<grid,block>>>(matrix_A_device,matrix_B_device,matrix_C_device,m,n,k);

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