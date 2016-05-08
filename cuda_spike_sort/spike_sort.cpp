#include <iostream>
#include <fstream>
#include <getopt.h>
#include <string>
#include "stdint.h"
#include <limits>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <pthread.h>
#include "work_queue.h"

#include "spike_sort.h"
#include "cycle_timer.h"
#include <cstdlib>

void printCudaInfo();
void spike_sort_init();
void find_spike_cluster();
void spline_alignment(double *, int);
void batch_interpolation(double *, int);
void print_cluster_info();
double* data;
int num_spike_cpu = 0;
bool done = false;
int num_spike_gpu = 0;

WorkQueue<double*> spike_datapoint;

void read_spike_dataset() {
    std::ifstream datafile;
    datafile.open(FILENAME, std::ios::in | std::ios::binary);
    if (datafile.is_open()) {
        data = new double[NUM_DATAPOINTS_PERCHANNEL*NUM_CHANNELS];
        datafile.read((char*)data,
                sizeof(double)*NUM_DATAPOINTS_PERCHANNEL*NUM_CHANNELS);
        datafile.close();
    } else {
        std::cout << " Not able to open dataset file : " << FILENAME << std::endl;
        return;
    }
#ifdef DEBUG_DATASET
    for (int i = 0; i < NUM_CHANNELS; i++) {
        std::cout << std::endl;
        std::cout << "channel " << i << " datapoints " << std::endl;
        for (int j = 0; j < WINDOW_SIZE; j++) {
            std::cout << data[i*NUM_DATAPOINTS_PERCHANNEL + j] << ", ";
        }
    }
    std::cout << std::endl;
#endif
}


inline double data_point(const double* data, const int channel, const int tp) {
    return ((double(*)[NUM_DATAPOINTS_PERCHANNEL])data)[channel][tp];
}

void* spike_detection_thread(void *arg) {
    std::cout << " in spike detection thread thrread " << std::endl;
    int iter = 0;
    int bt_cnt = 0;
    //double *spike_train = (double *)std::malloc(sizeof(double) * BATCH_SIZE * WINDOW_SIZE * NUM_CHANNELS);
    double *spike_train = (double *)std::malloc(sizeof(double) * BATCH_SIZE * WINDOW_SIZE * NUM_CHANNELS);
    //double *spike_train = (double *)std::malloc(sizeof(double) * WINDOW_SIZE * NUM_CHANNELS);
    //while ((iter + WINDOW_SIZE < NUM_DATAPOINTS_PERCHANNEL  && num_spike_cpu < 200)) {
    while ((iter + WINDOW_SIZE < NUM_DATAPOINTS_PERCHANNEL)) {
        if (data_point(data, 0 /* channel 0 */, iter) >= DETECT_THRESH) {
            //increment num_spikes
            int temp_iter = (iter - LEFT_WINDOW >= 0 )? iter - LEFT_WINDOW : iter;
	    //printf (" %d bt_cnt \n", bt_cnt); 
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                for (int tp = 0; tp < WINDOW_SIZE; tp++) {
                    spike_train[(bt_cnt * NUM_CHANNELS * WINDOW_SIZE) + (ch * WINDOW_SIZE) + tp] = data_point(data, ch, (temp_iter + tp));
                    //spike_train[(ch * WINDOW_SIZE) + tp] = data_point(data, ch, (temp_iter + tp));
                }
            }
            num_spike_cpu++;
	    bt_cnt++;
	    if (bt_cnt == BATCH_SIZE) {
	    //std::cout << "in the detect thread adding a spike " << std::endl; 
		    //printf(" %p adding spike train \n", spike_train);
		    spike_datapoint.put_work(spike_train);
		    //spike_train = (double *)std::malloc(sizeof(double) * WINDOW_SIZE * NUM_CHANNELS);
		    spike_train = (double *)std::malloc(sizeof(double) * BATCH_SIZE* WINDOW_SIZE * NUM_CHANNELS);
		    bt_cnt = 0;
		    //printf(" %p spike train  new ", spike_train);
		    if (num_spike_cpu == 20000) 
			break;
	    }
	    iter += WINDOW_SIZE;
        } else {
            iter++;
        }
    }
    std::cout << "num spikes detected thread" << num_spike_cpu << std::endl;
    done = true;
    return 0;
}

void* spike_handler_thread(void *arg) {
    std::cout << " in spike handler thrread " << std::endl;
    double temp = 0;
    while(!done || (num_spike_gpu != num_spike_cpu)) {
    //while(num_spike_gpu < 200) {
        double starttime = CycleTimer::currentSeconds();

        double *spike_train = spike_datapoint.get_work();
	//printf (" %p get work \n", spike_train);
        double endtime = CycleTimer::currentSeconds();
        temp += endtime - starttime;
        //std::cout << " count : "<< num_spike_gpu<< std::endl;
        batch_interpolation(spike_train, num_spike_gpu);
        spline_alignment(spike_train, num_spike_gpu);
        find_spike_cluster();
        num_spike_gpu += BATCH_SIZE;
        //num_spike_gpu ++;
	//printf (" %p get work free \n", spike_train);
	std::free(spike_train);
    }
    std::cout << "num spikes handler thread" << num_spike_gpu << std::endl;
    std::cout << "time spent in the queue : " << temp * 1000 << " ms " << std::endl;
    double temp_start = CycleTimer::currentSeconds();
    print_cluster_info();
    double temp_end = CycleTimer::currentSeconds();
    std::cout << "time to print" << (temp_end - temp_start) * 1000 << " ms " << std::endl;
    
    return 0;
}

int main() {

    read_spike_dataset();
    printCudaInfo();
#ifdef DEBUG_DATASET_ONLY
    std::cout << "calling cuda info" << std::endl;
    std::cout << "done calling cuda info" << std::endl;
    return 1;
#endif
    // creating spike detection thread.
    std::cout << "in main " << std::endl;
    spike_sort_init();
    double starttime = CycleTimer::currentSeconds();
    pthread_t tid1, tid2;
    int rc;
    rc = pthread_create(&tid1, NULL, spike_handler_thread, (void *) 6);
    if (rc) {
        assert(0);
    }
    rc = pthread_create(&tid2, NULL, spike_detection_thread, (void *)1);
    if (rc) {
        assert(0);
    }
    std::cout << "waiting for second thread" << std::endl;
    pthread_join(tid1, NULL);
    double endtime = CycleTimer::currentSeconds();
    std::cout << " total run time : " << (endtime - starttime) * 1000 << std::endl;
    delete data;
    return 0;
}
