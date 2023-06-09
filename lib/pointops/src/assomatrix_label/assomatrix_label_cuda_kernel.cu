#include "../cuda_utils.h"
#include "assomatrix_label_cuda_kernel.h"

// input: xyz (b, n, 3) new_xyz (b, m, 3)
// output: idx (b, m, nsample) dist2 (b, m, nsample)

// Note: n is the number of points, m is the number of clusters
// Note: nsample <= 1000

// input: idx_c (b, n, ks) lab (b, n, 1) cid (b, m, 1)
// output: idx (b, m, n) cnt (b, m, 1) clab (b, m, class)
__global__ void assomatrix_label_cuda_kernel(int b, int n, int m, int ks, const int *__restrict__ idx_c,  const int *__restrict__ cid, int *__restrict__ idx, int *__restrict__ cnt) {
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    //new_xyz += bs_idx * m * 3 + pt_idx * 3;
    //cid += bs_idx * m * 1 + pt_idx * 1;
    cid += bs_idx * m * 1 + pt_idx * 1;
    //xyz += bs_idx * n * 3;
    idx_c += bs_idx * n * ks;  // add

    idx += bs_idx * m * n + pt_idx * n;
    cnt += bs_idx * m * 1 + pt_idx * 1;     // count number of points located in one superpoint

    int cluster_id = cid[0];

    int num = 0;
    for(int k = 0; k < n; k++){
        for (int j = 0; j < ks; j++) {
            int id = idx_c[k * ks + j]; // cluster id of i-th point
            if (id == cluster_id) {
                //tmpi[k] = 1;
                idx[k] = 1;
                num++;
            }
        }
    }
    //for(int i = 0; i < n; i++){
    //    idx[i] = tmpi[i];
    //}
    cnt[0] = num;
    //delete []best;
    //delete []besti;
}


void assomatrix_label_cuda_launcher(int b, int n, int m, int ks, const int *idx_c, const int *cid, int *idx, int *cnt, cudaStream_t stream) {
    // param new_xyz: (B, m, 3)
    // param xyz: (B, n, 3)
    // param idx: (B, m, n)
    // param cnt: (B, m, 1)
    // param clab: (B, m, class)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    assomatrix_label_cuda_kernel<<<blocks, threads, 0, stream>>>(b, n, m, ks, idx_c, cid, idx, cnt);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
