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


//使用double buffer
#define FETCH_FLOAT4(pointer)(reinterpret_cast<float4 *>(&(pointer))[0])
template<const int BLOCK_SIZE_M,//每个BLOCK要计算的C的高度
         const int BLOCK_SIZE_N,//每个BLOCK加载进shared mem的A的宽度
         const int BLOCK_SIZE_K,//每个BLOCK要计算的C的宽度
         const int THREAD_SIZE_Y,//每个线程要计算的C的高度
         const int THREAD_SIZE_X//每个线程要计算的C的宽度
         >
__global__ void cuda_sgemm(float* A_ptr,float* B_ptr,float* C_ptr,const int M,int K,int N){
    int  tx=threadIdx.x;
    int  ty=threadIdx.y;

    int  bx=blockIdx.x;
    int  by=blockIdx.y;

    const int tid=ty*blockIdx.y+tx;

    //double buffer
    __shared__ float a_shared[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float b_shared[2][BLOCK_SIZE_K][BLOCK_SIZE_N];


    float accum[THREAD_SIZE_Y][THREAD_SIZE_X]={0.f};//定义寄存器
    float reg_a[THREAD_SIZE_Y]={0.f};
    float reg_b[THREAD_SIZE_X]={0.f};
    float ldg_a_reg{4}={0.f}//转置寄存

    float *A_ptr_start=A_ptr+blockIdx.y*BLOCK_SIZE_M*K;
    float *B_ptr_start=B_ptr+blockIdx.x*BLOCK_SIZE_N;

    const int A_tile_thread_per_row=BLOCK_SIZE_K/4;//2
    const int B_tile_thread_per_row=BLOCK_SIZE_N/4;//32

    const int A_tile_tid_x=tid%A_tile_thread_per_row;
    const int A_tile_tid_y=tid/A_tile_thread_per_row;
    const int B_tile_tid_y=tid%B_tile_thread_per_row;
    const int B_tile_tid_x=tid/B_tile_thread_per_row;

    FETCH_FLOAT4(ldg_a_reg[0])=FETCH_FLOAT4(A_ptr_start[K*A_tile_tid_y+A_tile_tid_x*4]);
    a_shared[0][A_tile_tid_x*4][A_tile_tid_y]=ldg_a_reg[0];
    a_shared[0][A_tile_tid_x*4+1][A_tile_tid_y]=ldg_a_reg[1];
    a_shared[0][A_tile_tid_x*4+2][A_tile_tid_y]=ldg_a_reg[2];
    a_shared[0][A_tile_tid_x*4+3][A_tile_tid_y]=ldg_a_reg[3];

    FETCH_FLOAT4(b_shared[0][B_tile_tid_y][B_tile_tid_x*4])=FETCH_FLOAT4(B_ptr_start[N*B_tile_tid_y+B_tile_tid_x*4]);
    __syncthreads();


    int stage_index=1;
    for(int s=BLOCK_SIZE_K;s<K;s+=BLOCK_SIZE_K){
       FETCH_FLOAT4(ldg_a_reg[0])=FETCH_FLOAT4(A_ptr_start[K*A_tile_tid_y+A_tile_tid_x*4+s]);
        a_shared[stage_index][A_tile_tid_x*4][A_tile_tid_y]=ldg_a_reg[0];
        a_shared[stage_index][A_tile_tid_x*4+1][A_tile_tid_y]=ldg_a_reg[1];
        a_shared[stage_index][A_tile_tid_x*4+2][A_tile_tid_y]=ldg_a_reg[2];
        a_shared[stage_index][A_tile_tid_x*4+3][A_tile_tid_y]=ldg_a_reg[3];

        FETCH_FLOAT4(b_shared[stage_index][B_tile_tid_y][B_tile_tid_x*4])=FETCH_FLOAT4(B_ptr_start[N*(B_tile_tid_y+s)+B_tile_tid_x*4]); 

        stage_index=stage_index^1;
        for(int k=0;k<BLOCK_SIZE_K;k++){

        FETCH_FLOAT4(reg_a[0])=FETCH_FLOAT4(a_shared[stage_index][k][ty*THREAD_SIZE_Y]);
        FETCH_FLOAT4(reg_a[4])=FETCH_FLOAT4(a_shared[stage_index][k][ty*THREAD_SIZE_Y+4]);
        FETCH_FLOAT4(reg_b[0])=FETCH_FLOAT4(b_shared[stage_index][k][tx*THREAD_SIZE_X]);
        FETCH_FLOAT4(reg_b[4])=FETCH_FLOAT4(b_shared[stage_index][k][tx*THREAD_SIZE_X+4]);

        for(int i=0;i<THREAD_SIZE_Y;i++)
            for(int j=0;j<THREAD_SIZE_X;j++)
            accum[i][j]+=reg_a[i]*b[j];
    }
            __syncthreads();
    }
    //最后一次计算
    stage_index=stage_index^1;
    for(int k=0;k<BLOCK_SIZE_K;k++){

        FETCH_FLOAT4(reg_a[0])=FETCH_FLOAT4(a_shared[stage_index][k][ty*THREAD_SIZE_Y]);
        FETCH_FLOAT4(reg_a[4])=FETCH_FLOAT4(a_shared[stage_index][k][ty*THREAD_SIZE_Y+4]);
        FETCH_FLOAT4(reg_b[0])=FETCH_FLOAT4(b_shared[stage_index][k][tx*THREAD_SIZE_X]);
        FETCH_FLOAT4(reg_b[4])=FETCH_FLOAT4(b_shared[stage_index][k][tx*THREAD_SIZE_X+4]);

        for(int i=0;i<THREAD_SIZE_Y;i++)
            for(int j=0;j<THREAD_SIZE_X;j++)
            accum[i][j]+=reg_a[i]*b[j];
    }
            __syncthreads();


    
    float *C_ptr_start=C_ptr+N*by*BLOCK_SIZE_M+bx*BLOCK_SIZE_N;
      for(int i=0;i<THREAD_SIZE_Y;i++){
            FETCH_FLOAT4([N*(ty*THREAD_SIZE_Y+i)+tx*THREAD_SIZE_X])=FETCH_FLOAT4(accum[i][0]);
            FETCH_FLOAT4([N*(ty*THREAD_SIZE_Y+i)+tx*THREAD_SIZE_X+4        ])=FETCH_FLOAT4(accum[i][4]);
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

    const int BLOCK_SIZE_M=128;
    const int BLOCK_SIZE_K=8;
    const int BLOCK_SIZE_N=128;
    const int THREAD_SIZE_X=8;
    const int THREAD_SIZE_Y=8;
    dim3 block(16,16);
    dim3 grid(m/BLOCK_SIZE_M,n/BLOCK_SIZE_N);
    
    cuda_sgemm
    <BLOCK_SIZE_M,
     BLOCK_SIZE_N,
     BLOCK_SIZE_K,
     THREAD_SIZE_X,
     THREAD_SIZE_Y>
     <<<grid,block>>>(matrix_A_device,matrix_B_device,matrix_C_device,m,n,k);

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