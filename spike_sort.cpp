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

bool temp_b = false;
double* data;
int spk_info[MAX_NUM_SPIKES];
cls_info spk_cls_state;
double alignment_time = 0;
double spline_interpolation_time = 0;
double clustering_algotime = 0;
double prune_time = 0;
int num_spike_cpu = 0;
bool done = false;
int num_spike_gpu = 0;

WorkQueue<double*> spike_datapoint;

void init_spk_info() {
    // init spk_cls_state TODO move it somewhere else
    spk_cls_state.num_spikes = 0;
    spk_cls_state.num_clusters = 1;

    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        for (int tp = 0; tp < NUM_INTERP_SAMPS; tp++) {
            spk_cls_state.spk_cls_mean[NOISE_CLUSTER][ch][tp] = 0;
        }
    }
}
void read_spike_dataset() {
    std::ifstream datafile;
    datafile.open(FILENAME, std::ios::in | std::ios::binary);
    if (datafile.is_open()) {
        data = new double[(long)NUM_DATAPOINTS_PERCHANNEL*(long)NUM_CHANNELS];
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
    double starttime = CycleTimer::currentSeconds();
    int iter = 0;
    temp_b = true;
    while ((iter + WINDOW_SIZE < NUM_DATAPOINTS_PERCHANNEL )) {
        if (data_point(data, 0 /* channel 0 */, iter) >= DETECT_THRESH) {
            //increment num_spikes
            num_spike_cpu++;
            int temp_iter = (iter - LEFT_WINDOW >= 0 )? iter - LEFT_WINDOW : iter;
            double *spike_train = new double[WINDOW_SIZE * NUM_CHANNELS];
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                for (int tp = 0; tp < WINDOW_SIZE; tp++) {
                    spike_train[(ch * WINDOW_SIZE) + tp] = data_point(data, ch, (temp_iter + tp));
                }
            }
            spike_datapoint.put_work(spike_train);
            //std::cout << " adding a spike to workqueue" << std::endl;
            iter += WINDOW_SIZE;
        } else {
            iter++;
        }
    }
    done = true;
    std::cout << "done detection  num spikes : " << num_spike_cpu << std::endl;
    double endtime = CycleTimer::currentSeconds();
    std::cout << "Total thread 1 runtime : " << (endtime - starttime) * 1000 << " ms " << std::endl;
    return arg;
}

void get_spline_values(const double* data, spline_quad* res) {
    double u[NUM_CHANNELS][WINDOW_SIZE + 1], z[NUM_CHANNELS][WINDOW_SIZE + 1];
    double p;

    //decomposition loop
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        u[ch][0] = 0;
        z[ch][0] = 0;
        z[ch][WINDOW_SIZE-1] = 0;
        z[ch][WINDOW_SIZE] = 0;

        for (int i = 1; i < WINDOW_SIZE; i++) {
            p = 0.5 * z[ch][i-1] + 2.0;
            z[ch][i] = -0.5/p;
            u[ch][i] = data[(ch *WINDOW_SIZE) + 1] + data[(ch * WINDOW_SIZE)+ i - 1] - 2 * data[(ch * WINDOW_SIZE)+ i];
            u[ch][i] = (3*u[ch][i] - 0.5*u[ch][i-1])/p;
        }

        //back substitution loop
        for (int i = WINDOW_SIZE; i > 0; i--) {
            z[ch][i] = z[ch][i] * z[ch][i+1] + u[ch][i];
        }
    }


    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        for (int i = 0; i < WINDOW_SIZE; i++) {
            res[ch*(WINDOW_SIZE)+i].xip1 = data[(ch * WINDOW_SIZE)+i+1];
            res[ch*(WINDOW_SIZE)+i].xi = data[(ch* WINDOW_SIZE) + i];
            res[ch*(WINDOW_SIZE)+i].zip1 = z[ch][i+1];
            res[ch*(WINDOW_SIZE)+i].zi = z[ch][i];
        }
    }
}

double spline_eval_point(double t, spline_quad* sqs, int ch) {
    assert(t < WINDOW_SIZE);

    int f = t;
    if (f == WINDOW_SIZE)
        f--;

    double P0 = sqs[(ch * WINDOW_SIZE) + f].xip1;
    double P1 = sqs[(ch * WINDOW_SIZE)+ f].xi;
    double P2 = sqs[(ch * WINDOW_SIZE)+ f].zip1;
    double P3 = sqs[(ch * WINDOW_SIZE)+ f].zi;
    double a = t - ((double) f);
    double b = 1 - a;

    return P0*a + P1*b + P2*(a*a*a-a)/6 + P3*(b*b*b-b)/6;
}

void closest_cluster(double* spike, cluster_res* res) {
    double min = std::numeric_limits<double>::max();
    int minc = NOISE_CLUSTER;

    for (int i = 0; i < spk_cls_state.num_clusters; i++) {
        double d = 0;
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            for (int tp = 0; tp < NUM_INTERP_SAMPS; tp++) {
                double dd =  spike[ch*NUM_INTERP_SAMPS+tp] - spk_cls_state.spk_cls_mean[i][ch][tp];
                d += dd*dd;
            }
        }
        if (d < min) {
            min = d;
            minc = i;
        }
    }
    res->clusterId = minc;
    res->dist = min;
}

void cluster_spike(double* spline_spike, int *spk_info) {
    spk_cls_state.num_spikes++;
    cluster_res cr;
    closest_cluster(spline_spike, &cr);
    if (spk_cls_state.spk_cls_size[cr.clusterId] == 0) {
        cr.clusterId = NOISE_CLUSTER;
    }
    //std::cout << "cluster id " << cr.clusterId << "     " << " clsuter dist : " << cr.dist << std::endl;

    //if certain enough
    if (cr.dist < CERTAINTY_CUTOFF) {
        //assign to cluster km
        spk_info[spk_cls_state.num_spikes - 1] = cr.clusterId;
        //increment size of cluster km
        double clsz = (double) ++spk_cls_state.spk_cls_size[cr.clusterId];

        //update cluster mean (PROBABLY PARALLEL? MAYBE JUST OMP?)
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            for (int tp = 0; tp < NUM_INTERP_SAMPS; tp++) {
                spk_cls_state.spk_cls_mean[cr.clusterId][ch][tp] =
                    (((clsz-1) * spk_cls_state.spk_cls_mean[cr.clusterId][ch][tp]) + spline_spike[ch*NUM_INTERP_SAMPS+tp])/clsz;
            }
        }
    } else {
        //make new cluster
        cr.clusterId = spk_cls_state.num_clusters;
        //increment num_clusters
        spk_cls_state.num_clusters++;
        //set this as mean
        for (int i = 0; i < NUM_CHANNELS; i++) {
            for (int j = 0; j < NUM_INTERP_SAMPS; j++) {
                spk_cls_state.spk_cls_mean[cr.clusterId][i][j] = spline_spike[i*NUM_INTERP_SAMPS+j];
            }
        }
        //set size of cluster to 1
        spk_cls_state.spk_cls_size[cr.clusterId] = 1;
        //assign spike to new cluster
        spk_info[spk_cls_state.num_spikes - 1] = cr.clusterId;
    }

    //spikes, then we prune it here
    int lookback_spike = spk_cls_state.num_spikes - LOOKBACK_DIST;
    if (lookback_spike >= 0) {
        int lookback_cluster = spk_info[lookback_spike];
        int lookback_size = spk_cls_state.spk_cls_size[lookback_cluster];
        //if cluser_size[spike[num_spikes - lookback_dist].cluster] < min_cluster_thresh
        if (lookback_size < MIN_CLUSTER_THRESH) {
            //assign the num_spikes - lookback_dist data point to noise cluster
            spk_info[lookback_spike] = PRUNED_CLUSTER;

            //if spk_cls_size == 0  it's already pruned
            if (lookback_size != 0) {
                //prune cluster (set cluster mean to 0, anything else?)
                for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                    for (int tp = 0; tp < NUM_INTERP_SAMPS; tp++) {
                        spk_cls_state.spk_cls_mean[lookback_cluster][ch][tp] = 0;
                    }
                }
                //this spk_cls_size = 0 (this is our flag so we know it's been pruned)
                spk_cls_state.spk_cls_size[lookback_cluster] = 0;
            }
        }
    }
}

void eliminate_noise_clusters() {
    //prune last LOOKBACK_DIST clusters after while loop has finished
    int cl;
    int sz;
    for (int i = spk_cls_state.num_spikes - LOOKBACK_DIST + 1; i < spk_cls_state.num_spikes; i++) {
        cl = spk_info[i];
        sz = spk_cls_state.spk_cls_size[cl];
        if (sz < MIN_CLUSTER_THRESH) {
            spk_info[i] = PRUNED_CLUSTER;
            if (sz != 0) {
                for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                    for (int tp = 0; tp < NUM_INTERP_SAMPS; tp++) {
                        spk_cls_state.spk_cls_mean[cl][ch][tp] = 0;
                    }
                }
                spk_cls_state.spk_cls_size[cl] = 0;
            }
        }
    }
}

void spike_alignment(double *sp) {
    int max_index[NUM_CHANNELS];
    double max_value[NUM_CHANNELS];
    // calculate max_index in each channel
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        max_value[ch] = std::numeric_limits<double>::min();
        for (int tp = 0; tp < NUM_INTERP_SAMPS; tp++) {
            if (sp[(ch * NUM_INTERP_SAMPS) + tp] > max_value[ch]) {
                max_index[ch] = tp;
                max_value[ch] = sp[(ch * NUM_INTERP_SAMPS) + tp];
            }
        }
    }
    // take average of the max_index and align the max to that index in
    // each channel
    int average_max_index = 0;
    for (int i = 0 ; i < NUM_CHANNELS; i++) {
        average_max_index += max_index[i];
    }
    average_max_index /= NUM_CHANNELS;

    //align each channel with respect to max index.
    for (int i = 0; i < NUM_CHANNELS; i++) {
        int idx_diff = average_max_index - max_index[i];
        if (idx_diff > 0) {
            for (int j = NUM_INTERP_SAMPS; j >= 0; j--) {
                if (j - idx_diff >= 0)
                    sp[j] = sp[j - idx_diff];
                else
                    sp[j] = 0;
            }
        } else {
            for (int j = 0; j < NUM_INTERP_SAMPS; j++) {
                if (j + std::abs(idx_diff) < NUM_INTERP_SAMPS)
                    sp[j] = sp[j + std::abs(idx_diff)];
                else
                    sp[j] = 0;
            }
        }
    }
}

void calculate_spline_interpolation(double *spline_spike, double *spike_train) {
    spline_quad sqs[NUM_CHANNELS*WINDOW_SIZE];
    get_spline_values(spike_train, sqs);
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        for (int tp = 0; tp < NUM_INTERP_SAMPS; tp++) {
            spline_spike[ch*NUM_INTERP_SAMPS+tp] = spline_eval_point(((double)tp)*((double)WINDOW_SIZE)/((double)NUM_INTERP_SAMPS), sqs, ch);
        }
    }
}

void print_cluster_info() {
    int total_spike_count = 0;
    std::cout<<"cluster result: " << std::endl;
    for (int i = 0; i < spk_cls_state.num_clusters; i++) {
        std::cout<< "cluster " << i << " :" << spk_cls_state.spk_cls_size[i] << std::endl;
        total_spike_count += spk_cls_state.spk_cls_size[i];
    }
    std::cout << std::endl << "Total spikes detected : " << total_spike_count << std::endl;
}

void* spike_handler_thread(void *arg) {
    double temp = 0;
    init_spk_info();
    while(!temp_b);
    std::cout << "in the spike handler thread " << std::endl;
    while(!done || (num_spike_gpu != num_spike_cpu)) {
        double starttime = CycleTimer::currentSeconds();
        double *spike_train = spike_datapoint.get_work();
        double endtime = CycleTimer::currentSeconds();
        temp  += endtime - starttime;
        // spline interpolation
        //
        //
        //
        //
        double start_spline_time = CycleTimer::currentSeconds();
        double spline_spike[NUM_INTERP_SAMPS*NUM_CHANNELS];
        calculate_spline_interpolation(spline_spike, spike_train);
        double end_spline_time = CycleTimer::currentSeconds();
        spline_interpolation_time += (end_spline_time - start_spline_time);
#ifdef DEBUG_SPLINE
        for (int i = 0; i < NUM_CHANNELS; i++) {
            std::cout << "CHANNEL " << i << " :" << std::endl << std::endl;
            std::cout << "Actual spike channel " << i << std::endl;
            for (int j = 0; j < WINDOW_SIZE; j++) {
                std::cout << spike_train[j] << " ";
            }
            std::cout << std::endl <<std::endl;
            std::cout << "interpolated spike channel " << i << std::endl;
            for(int j = 0; j < NUM_INTERP_SAMPS; j++) {
                std::cout << spline_spike[j] << " ";
            }
            std::cout << std::endl <<std::endl;
        }
#endif

        // do spike_alignment
        //
        //
        //
        //
        double start_alignment_time = CycleTimer::currentSeconds();
        spike_alignment(spline_spike);
        double end_alignment_time = CycleTimer::currentSeconds();
        alignment_time += (end_alignment_time - start_alignment_time);


        // do clustering
        //
        //
        //
        //
        //
        double start_clustering_time = CycleTimer::currentSeconds();
        cluster_spike(spline_spike, spk_info);
        double end_clustering_time = CycleTimer::currentSeconds();
        clustering_algotime += (end_clustering_time - start_clustering_time);
        num_spike_gpu++;
        delete spike_train;
    }
    std::cout << "done clustering spike" << std::endl;
    std::cout << "Time spent in queue: " << temp * 1000 << " ms " << std::endl;
    double start_prune_time = CycleTimer::currentSeconds();
    eliminate_noise_clusters();
    print_cluster_info();
    double end_prune_time = CycleTimer::currentSeconds();
    prune_time += (end_prune_time - start_prune_time);
    return arg;
}

int main() {
    read_spike_dataset();
#ifdef DEBUG_DATASET_ONLY
    return 1;
#endif
    // creating spike detection thread.
    std::cout << "in main " << std::endl;
    pthread_t tid1, tid2;
    double starttime = CycleTimer::currentSeconds();
    int rc;
    rc = pthread_create(&tid2, NULL, spike_detection_thread, (void *)1);
    if (rc) {
        assert(0);
    }
    rc = pthread_create(&tid1, NULL, spike_handler_thread, (void *) 6);
    if (rc) {
        assert(0);
    }
    std::cout << "waiting for second thread" << std::endl;
    pthread_join(tid1, NULL);
    double endtime = CycleTimer::currentSeconds();
    std::cout << "Total runtime : " << (endtime - starttime) * 1000 << " ms " << std::endl;
    std::cout << "cubic spline time : " << spline_interpolation_time * 1000 << " ms " << std::endl;
    std::cout << "alignment time : " <<  alignment_time * 1000 << " ms " << std::endl;
    std::cout << "clustering time : " << clustering_algotime * 1000 << " ms " << std::endl;
    std::cout << "prune time : " << prune_time * 1000 << " ms " << std::endl;
    delete data;
    return 0;
}
