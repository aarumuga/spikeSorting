#include <iostream>
#include <limits>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include "spike_sort.h"
#include "cycle_timer.h"

#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<driver_functions.h>
double create_train;
double *gpu_spike_train;
double *spline_quad_vector;
double *spline_U;
double *spline_Z;
double *gpu_inter_spike_train;
double *spike_sort_clusters;
double cpy_time;
int *spike_max_index;
int *cluster_spike_count;
int *num_clusters;
int num_cluster_cpu = 1;
double *closest_cluster_dist;
double *cluster_result;
int spike_result[MAX_NUM_SPIKES];
double res[2];
int res_cluster_count;
int count;
double interpolate_time;
double alignment_time;
double clustering_time;
double find_closest_time;
double relative_dist_time;
double update_cluster_time;
double update_clsz_time;
double res_time;
double memset_time;
#define CLUSTER_ID 0
#define CLUSTER_DIST 1
#define CLUSTER_RES 2


void CUDAMALLOC(int line, void **ptr, size_t size) {
	cudaMalloc(ptr, size);
        cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf(" line %d Error : %s \n ", line , cudaGetErrorString(err));
		assert(0);
	}
}

void CUDAMEMCPY(int line, void *dst, const void *src, size_t size, enum cudaMemcpyKind kind) {
	cudaMemcpy(dst, src, size, kind);
        cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf(" line %d Error : %s \n ", line , cudaGetErrorString(err));
		assert(0);
	}
}

void CUDAMEMSET(int line, void *devptr, int value, size_t count) {
      cudaError_t err = cudaMemset(devptr, value, count);
	if (err != cudaSuccess) {
		printf(" line %d Error : %s \n ", line , cudaGetErrorString(err));
		assert(0);
	}
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<double>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}

void spike_sort_init() {
    std::cout << " in gpu init " << std::endl;
    //memory for actual spike train
    CUDAMALLOC(__LINE__, (void **)&gpu_spike_train,
                sizeof(double) * (BATCH_SIZE * WINDOW_SIZE * NUM_CHANNELS));
    //memory for spline quad vector
    CUDAMALLOC(__LINE__, (void **)&spline_quad_vector,
                sizeof(double) * (BATCH_SIZE * (WINDOW_SIZE) * (NUM_CHANNELS) * 4));
    //memory for spline calculation U
    CUDAMALLOC(__LINE__, (void **)&spline_U,
                sizeof(double) * (BATCH_SIZE * (WINDOW_SIZE + 2) * NUM_CHANNELS));
    //memory for spline calculation Z
    CUDAMALLOC(__LINE__, (void **)&spline_Z,
                sizeof(double) * (BATCH_SIZE * (WINDOW_SIZE + 2) * NUM_CHANNELS));
    //memory for interpolate spline
    CUDAMALLOC(__LINE__, (void **)&gpu_inter_spike_train,
                sizeof(double) * BATCH_SIZE * (NUM_INTERP_SAMPS * NUM_CHANNELS));
    //memory for cluster
    CUDAMALLOC(__LINE__, (void **)&spike_sort_clusters,
                sizeof(double) * (NUM_INTERP_SAMPS * NUM_CHANNELS * MAX_NUM_CLUSTER));
    CUDAMALLOC(__LINE__, (void **)&spike_max_index,
                sizeof(int) * BATCH_SIZE * (NUM_CHANNELS + 1));
    CUDAMALLOC(__LINE__, (void **) &cluster_spike_count, sizeof(int) * (MAX_NUM_CLUSTER + 1));
    CUDAMALLOC(__LINE__, (void **) &num_clusters, sizeof(int));
    CUDAMALLOC(__LINE__, (void **) &closest_cluster_dist, sizeof(double) * MAX_NUM_CLUSTER);
    CUDAMALLOC(__LINE__, (void **) &cluster_result, sizeof(double) * 3); 
    int init_cluster_count = 1;
    CUDAMEMCPY(__LINE__, num_clusters, &init_cluster_count, sizeof(int), cudaMemcpyHostToDevice);
    //init_cluster_count = 10;
    //CUDAMEMCPY(__LINE__, &init_cluster_count, num_clusters,  sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "printing intial count : " << init_cluster_count << std::endl;
    CUDAMEMSET(__LINE__, cluster_spike_count, 0 , sizeof(int) * (MAX_NUM_CLUSTER + 1));
    CUDAMEMSET(__LINE__, spike_sort_clusters, 0, sizeof(double) * (NUM_INTERP_SAMPS * NUM_CHANNELS * MAX_NUM_CLUSTER));
    CUDAMEMSET(__LINE__, spline_U, 0, sizeof(double) *BATCH_SIZE * (NUM_CHANNELS * (WINDOW_SIZE + 2)));
    CUDAMEMSET(__LINE__, spline_Z, 0, sizeof(double) *BATCH_SIZE * (NUM_CHANNELS * (WINDOW_SIZE + 2)));
    std::cout << " done with allocation " << std::endl;
}

__global__ void calculate_spline_quad_vector(double *spk_train, double *u, double*z, double *res) {
        int ch = threadIdx.x * WINDOW_SIZE;
	int spk_idx = blockIdx.x * (WINDOW_SIZE + 2) * NUM_CHANNELS;
	int res_idx = blockIdx.x * (WINDOW_SIZE) * NUM_CHANNELS;
        u[spk_idx + ch + 0] = 0;
        z[spk_idx + ch + 0] = 0;
        z[spk_idx + ch + WINDOW_SIZE - 1] = 0;
        z[spk_idx + ch + WINDOW_SIZE] = 0;
    for (int i = 1; i < WINDOW_SIZE; i++) {
        double p = 0.5 * z[spk_idx + ch + i - 1] + 2.0;
        z[spk_idx + ch + i] = -0.5 / p;
        u[spk_idx + ch + i] = spk_train[res_idx + ch + 1] + spk_train[res_idx + ch + i - 1] - 2 * spk_train[res_idx + ch + i];
        u[spk_idx + ch + i] = ((3 * u[spk_idx + ch + i]) - (0.5 * u[spk_idx + ch + i - 1])) / p;
    }
    z[spk_idx + ch + WINDOW_SIZE + 1] = 0;

    for (int i = WINDOW_SIZE; i > 0; i--) {
        z[spk_idx + ch + i] = z[spk_idx + ch + i] * z[spk_idx + ch + i + 1] + u[spk_idx + ch + i];
    }
    z[spk_idx + ch + WINDOW_SIZE + 1] = 0;

    for (int i = 0; i < WINDOW_SIZE; i++) {
        res[((res_idx + ch + i) * 4)] = spk_train[res_idx + ch + i + 1];
        res[((res_idx + ch + i) * 4) + 1] = spk_train[res_idx + ch + i];
        res[((res_idx + ch + i) * 4) + 2] = z[spk_idx + ch + i + 1];
        res[((res_idx + ch + i) * 4) + 3] = z[spk_idx + ch + i];
    }
}

__global__ void get_interpolated_spike_train(double *gpu_inter_spike_train,
                                             double *spline_quad_vector) {
        int ch = threadIdx.x * NUM_INTERP_SAMPS;
	int spk_idx = blockIdx.x * (WINDOW_SIZE) * NUM_CHANNELS + (threadIdx.x * WINDOW_SIZE);
	double fact = ((double)WINDOW_SIZE / (double) NUM_INTERP_SAMPS);
	for (int i = 0; i < NUM_INTERP_SAMPS; i++) {
		double t = i * fact;
		int f = t;
		int quad_idx = (spk_idx + f) * 4;
		if (f == WINDOW_SIZE)
			f--;
		double P0 = spline_quad_vector[quad_idx + 0];
		double P1 = spline_quad_vector[quad_idx + 1];
		double P2 = spline_quad_vector[quad_idx + 2];
		double P3 = spline_quad_vector[quad_idx + 3];
		double a = t - ((double)f);
		double b = 1 - a;
		gpu_inter_spike_train[(blockIdx.x * NUM_INTERP_SAMPS * NUM_CHANNELS) + ch + i] = (P0 * a) + (P1 * b) + ((P2 *((a * a * a) - a)) / 6) + ((P3 * ((b * b * b) - b)) / 6);
	}
}

__global__ void spike_alignment_max_index_calculation(double *gpu_inter_spike_train,
        int *spike_max_index) {
    int ch = threadIdx.x * NUM_INTERP_SAMPS;
    double max_value = 0;
    for (int tp = 0; tp < NUM_INTERP_SAMPS; tp++) {
        if(gpu_inter_spike_train[ch + tp] > max_value) {
           spike_max_index[threadIdx.x] = tp;
        }
    }
    __syncthreads();
    if (threadIdx.x == 1) {
        int average_max_index = 0;
        for (int i = 0; i < NUM_CHANNELS; i++) {
            average_max_index += spike_max_index[i];
        }
        average_max_index /= NUM_CHANNELS;
        spike_max_index[NUM_CHANNELS] = average_max_index;
    }
    __syncthreads();
    int idx_diff = spike_max_index[NUM_CHANNELS] - spike_max_index[threadIdx.x];
    if (idx_diff > 0) {
        for (int j = NUM_INTERP_SAMPS; j >= 0; j--) {
            if (j - idx_diff >= 0) {
                gpu_inter_spike_train[ch + j] = gpu_inter_spike_train [ch + (j - idx_diff)];
            }else {
                gpu_inter_spike_train[ch + j] = 0;
            }
        }
    } else {
        idx_diff = -idx_diff;
        for (int i = 0; i < NUM_INTERP_SAMPS; i++) {
            if ( (i + idx_diff ) < NUM_INTERP_SAMPS) {
                gpu_inter_spike_train[ch + i] = gpu_inter_spike_train[ch + (i + idx_diff)];
            } else {
                gpu_inter_spike_train[ch + i] = 0;
            }
        }
    }
}
__global__ void find_closest_cluster(int *num_clusters,
                                     double *closest_cluster_dist,
                                     double *temp, int *cluster_spike_count) {
    temp[CLUSTER_ID] = NOISE_CLUSTER;
    temp[CLUSTER_DIST] = closest_cluster_dist[0];
    for (int i = 1; i < *num_clusters; i++) {
        if (closest_cluster_dist[i] < temp[CLUSTER_DIST]) {
            temp[CLUSTER_ID] = i;
            temp[CLUSTER_DIST] = closest_cluster_dist[i];
        }
    }
   
    if (temp[CLUSTER_DIST] < CERTAINTY_CUTOFF) {
	temp[CLUSTER_RES] = 0;
        cluster_spike_count[(int)temp[CLUSTER_ID]]++;
    } else {
	temp[CLUSTER_RES] = 1;
        (*num_clusters)++;
    }
}

__global__ void calculate_cluster_spike_relative_dist(double *gpu_inter_spike_train,
						      double *spike_sort_clusters,
                                                      double *closest_cluster_dist, int *num_clusters) {
	if (blockIdx.x < *num_clusters) {
		__shared__ double dist_diff_sq[NUM_CHANNELS];
		int ch_idx = threadIdx.x * NUM_INTERP_SAMPS;
		int cluster_idx = (blockIdx.x * NUM_CHANNELS * NUM_INTERP_SAMPS)+ ch_idx;
		double ch_accumulated_diff = 0;
		for (int i = 0; i < NUM_INTERP_SAMPS; i++) {
			double dd = gpu_inter_spike_train[ch_idx + i] - spike_sort_clusters[cluster_idx + i];
		        ch_accumulated_diff += dd * dd;
 		}
		dist_diff_sq[threadIdx.x] = ch_accumulated_diff;
		__syncthreads();
		int total_threads = NUM_CHANNELS;
		while(total_threads > 1) {
			int haltpoint = total_threads >> 1;
			if (threadIdx.x < haltpoint) {
				int second_idx = threadIdx.x + haltpoint;
				if (second_idx < NUM_CHANNELS * NUM_INTERP_SAMPS) {
					dist_diff_sq[threadIdx.x] += dist_diff_sq[second_idx];
				}
			}
			__syncthreads();
			total_threads = total_threads >> 1;
		}
		//__syncthreads();

		if (threadIdx.x == 0) {
			closest_cluster_dist[blockIdx.x] = dist_diff_sq[0];
		}
		//__syncthreads();
	}
}


__global__ void print_interpolated_kernel(double *gpu_inter_spike_train) {
	printf("printing form the kernel \n");
	for (int i = 0; i < NUM_CHANNELS; i++) {
		printf("interpolated spike channel %d \n", i); 
		for (int j = 0; j < NUM_INTERP_SAMPS; j++) {
			printf("%f ", gpu_inter_spike_train[(i * NUM_INTERP_SAMPS) + j]); 
		}
		printf("\n");
		printf("\n");
	}
		printf("\n");
		printf("\n");
		printf("\n");
		printf("\n");
}
__global__ void print_closest_cluseter_dist(double *closest_cluster_dist, int *num_clusters) {
          printf (" current num cluster = %d \n", *num_clusters);
	  for (int i = 0; i < *num_clusters; i++) {
		printf("cluster distance from %d cluster = %f\n", i, closest_cluster_dist[i]);
	  }
}
__global__ void update_cluster_mean_or_create_new_cluster(double *gpu_inter_spike_train,
							  double *spike_sort_clusters,
						   	  int *num_clusters,
							  double *cluster_result,
							  int *cluster_spike_count) {
    if ((int)cluster_result[CLUSTER_RES] == 0) {
            int clid = (int)cluster_result[CLUSTER_ID];
	    int cluster_size = cluster_spike_count[clid];
	    int spk_cl_offset = (clid * NUM_CHANNELS * NUM_INTERP_SAMPS) + (blockIdx.x * NUM_INTERP_SAMPS)+ threadIdx.x;
	    int spk_offset = (blockIdx.x * NUM_INTERP_SAMPS) + threadIdx.x;

	    spike_sort_clusters[spk_cl_offset] = (((cluster_size - 1)  * spike_sort_clusters[spk_cl_offset]) + gpu_inter_spike_train[spk_offset])/ (cluster_size);
    } else if ((int) cluster_result[CLUSTER_RES] == 1) {
            //if (blockIdx.x == 0 && threadIdx.x == 0)
	    //printf("creating new cluster \n");
            int clsiz = (*num_clusters - 1) ;
	    int clidx = (clsiz * NUM_CHANNELS * NUM_INTERP_SAMPS) + (blockIdx.x * NUM_INTERP_SAMPS) + threadIdx.x;
	    int spk_idx = (blockIdx.x * NUM_INTERP_SAMPS )+ threadIdx.x;
	    spike_sort_clusters[clidx] = gpu_inter_spike_train[spk_idx];
	    cluster_spike_count[clsiz] = 1;
    }
}

void batch_interpolation(double *cpu_spike_train, int spike_num) {
	cudaError_t err;
	double starttime = CycleTimer::currentSeconds();
	CUDAMEMCPY(__LINE__, gpu_spike_train, cpu_spike_train,
			sizeof(double) *BATCH_SIZE * (NUM_CHANNELS * WINDOW_SIZE),
			cudaMemcpyHostToDevice);
	calculate_spline_quad_vector<<< BATCH_SIZE, NUM_CHANNELS>>>(gpu_spike_train, spline_U, spline_Z, spline_quad_vector);
	err = cudaThreadSynchronize();
	if (err != cudaSuccess) {
		printf("Error 15 : %s \n ", cudaGetErrorString(err));
	}
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error 15 : %s \n ", cudaGetErrorString(err));
	}
	get_interpolated_spike_train<<<BATCH_SIZE, NUM_CHANNELS>>>(gpu_inter_spike_train, spline_quad_vector);
	err = cudaThreadSynchronize();
	if (err != cudaSuccess) {
		printf("Error 15 : %s \n ", cudaGetErrorString(err));
	}
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error 15 : %s \n ", cudaGetErrorString(err));
	}
	double endtime = CycleTimer::currentSeconds();
	alignment_time += endtime - starttime;
}

void spline_alignment(double *cpu_spike_train, int spike_num) {
	int bt_cnt = 0;
	cudaError_t err; 
	double starttime = CycleTimer::currentSeconds();
	while (bt_cnt < BATCH_SIZE) {
		double *spk_train = (double *)gpu_spike_train + (bt_cnt * NUM_CHANNELS * WINDOW_SIZE);
		double *inter_spk_train = (double *)gpu_inter_spike_train + (bt_cnt * NUM_CHANNELS * NUM_INTERP_SAMPS);
		double *quad = (double *)spline_quad_vector + (bt_cnt * NUM_CHANNELS * (WINDOW_SIZE) * 4);
                int *spk_midx = (int *)spike_max_index + (bt_cnt * (NUM_CHANNELS + 1));
		double temp_start = CycleTimer::currentSeconds();
		//print_interpolated_kernel<<<1, 1>>>(gpu_inter_spike_train);
#ifdef DEBUG_SPLINE
		//std::cout << "debug here " << std::endl;
		double interpolation_test[NUM_CHANNELS * NUM_INTERP_SAMPS];
		double cpu_test[NUM_CHANNELS * WINDOW_SIZE];
		CUDAMEMCPY(__LINE__, &interpolation_test, inter_spk_train,  sizeof(interpolation_test), cudaMemcpyDeviceToHost);
		CUDAMEMCPY(__LINE__, &cpu_test, spk_train, sizeof(cpu_test), cudaMemcpyDeviceToHost);
		for (int i = 0; i < NUM_CHANNELS; i++) {
			std::cout << "Actual spike from ch " << i << " window size : " << WINDOW_SIZE << std::endl;
			for ( int j = 0; j < WINDOW_SIZE; j++) {
				std::cout << cpu_test[(i * WINDOW_SIZE) + j] << " ";
			}
			std::cout << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;

			std::cout << "interpolated spike for ch " << i << " window size : " << NUM_INTERP_SAMPS << std::endl;
			for (int j = 0; j < NUM_INTERP_SAMPS; j++) {
				std::cout << interpolation_test[(i * NUM_INTERP_SAMPS) + j] << " ";
			}
			std::cout << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
		}
			std::cout << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
#endif
#ifdef DEBUG_ALIGNMENT
		spike_alignment_max_index_calculation<<<1, NUM_CHANNELS>>>(inter_spk_train, spk_midx);
		err = cudaThreadSynchronize();
		if (err != cudaSuccess) {
			printf("Error 17 : %s \n ", cudaGetErrorString(err));
		}
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Error 17 : %s \n ", cudaGetErrorString(err));
		}
		double alignment_test[NUM_CHANNELS * NUM_INTERP_SAMPS];
		CUDAMEMCPY(__LINE__, &alignment_test, inter_spk_train,  sizeof(alignment_test), cudaMemcpyDeviceToHost);
		for (int i = 0; i < NUM_CHANNELS; i++) {
			std::cout << "interpolated spike for ch " << i << " window size : " << NUM_INTERP_SAMPS << std::endl;
			for (int j = 0; j < NUM_INTERP_SAMPS; j++) {
				std::cout << alignment_test[(i * NUM_INTERP_SAMPS) + j] << " ";
			}
			std::cout << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
		}
#endif
		bt_cnt++;
	}
	err = cudaThreadSynchronize();
	if (err != cudaSuccess) {
		printf("Error 17 : %s \n ", cudaGetErrorString(err));
	}
	double endtime = CycleTimer::currentSeconds();
	interpolate_time += endtime - starttime;
}

void find_spike_cluster() {
	int bt_cnt = 0;
	while (bt_cnt < BATCH_SIZE) {
		double *inter_spk_train = (double *)gpu_inter_spike_train + (bt_cnt * NUM_CHANNELS * NUM_INTERP_SAMPS);
		double starttime = CycleTimer::currentSeconds();
		double temp_start = CycleTimer::currentSeconds();
		//num_blocks = num_cluster_cpu;
		calculate_cluster_spike_relative_dist<<<MAX_NUM_CLUSTER, NUM_CHANNELS>>>(inter_spk_train, spike_sort_clusters, closest_cluster_dist, num_clusters);
		cudaError_t err = cudaThreadSynchronize();
		if (err != cudaSuccess) {
			printf("Error 13 : %s \n ", cudaGetErrorString(err));
		}
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf(" %d Error 18 : %s \n ", __LINE__ ,cudaGetErrorString(err));
		}
		double temp_end = CycleTimer::currentSeconds();
		relative_dist_time += temp_end - temp_start;
		//print_closest_cluseter_dist<<<1, 1>>>(closest_cluster_dist, num_clusters);
		temp_start = CycleTimer::currentSeconds();
		find_closest_cluster<<<1, 1>>>(num_clusters, closest_cluster_dist, cluster_result, cluster_spike_count);
		err = cudaThreadSynchronize();
		if (err != cudaSuccess) {
			printf("Error 1 : %s \n ", cudaGetErrorString(err));
		}
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Error 1 : %s \n ", cudaGetErrorString(err));
		}
		temp_end = CycleTimer::currentSeconds();
		find_closest_time += temp_end - temp_start;
		temp_start = CycleTimer::currentSeconds();
		update_cluster_mean_or_create_new_cluster<<<NUM_CHANNELS, NUM_INTERP_SAMPS>>>(inter_spk_train,
				spike_sort_clusters, num_clusters, cluster_result, cluster_spike_count);
		err = cudaThreadSynchronize();
		if (err != cudaSuccess) {
			printf("Error 19 : %s \n ", cudaGetErrorString(err));
		}
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Error 19 : %s \n ", cudaGetErrorString(err));
		}
		temp_end = CycleTimer::currentSeconds();
		update_cluster_time += temp_end - temp_start;
		double endtime = CycleTimer::currentSeconds();
		clustering_time += endtime - starttime;
		bt_cnt++;
	}
}

void print_cluster_info() {
    int cl_sp_cnt[MAX_NUM_CLUSTER + 1];
    int total_clusters;
    CUDAMEMCPY(__LINE__, &cl_sp_cnt, cluster_spike_count, sizeof(int) * (MAX_NUM_CLUSTER + 1), cudaMemcpyDeviceToHost);
    CUDAMEMCPY(__LINE__, &total_clusters, num_clusters, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "total_clusters " << total_clusters << std::endl;
    int total_spike_count = 0;
    std::cout<<"cluster result: " << std::endl;
    for (int i = 0; i < total_clusters; i++) {
        std::cout<< "cluster " << i << " :" << cl_sp_cnt[i] << std::endl;
        total_spike_count += cl_sp_cnt[i];
    }
    std::cout << std::endl << "Total spikes detected : " << total_spike_count << std::endl;
    std::cout << "haha" << std::endl;
    std::cout << "interpolation time : " << interpolate_time * 1000 << " ms " << std::endl;
    std::cout << "alignment_time : " << alignment_time * 1000 << " ms " << std::endl;
    std::cout << "clustering_time : " << clustering_time * 1000 << " ms " << std::endl;
    std::cout << "relative_dist_time  : " << relative_dist_time * 1000 << " ms " << std::endl;
    std::cout << "find_closest_time : " << find_closest_time * 1000 << " ms " << std::endl;
    std::cout << "update_cluster_time : " << update_cluster_time * 1000 << " ms " << std::endl;
    std::cout << "memsettime  : " << memset_time * 1000 << " ms " << std::endl;
    std::cout << "res time : " << res_time * 1000 << " ms " << std::endl;
    std::cout << "update_clsz_time  : " << update_clsz_time * 1000 << " ms " << std::endl;
    std::cout << "create time  : " << create_train * 1000 << " ms " << std::endl;
}
