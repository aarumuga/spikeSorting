#define WINDOW_SIZE 20 //window in samples
#define LEFT_WINDOW 0
#define DETECT_THRESH 0.5f //TODO
#define NOISE_CLUSTER 0
#define PRUNED_CLUSTER 0
#define CERTAINTY_CUTOFF 0.1f //TODO
#define MIN_CLUSTER_THRESH 5 //TODO
#define LOOKBACK_DIST 100 //TODO
#define MAX_NUM_CLUSTER 1024
#define NUM_CHANNELS 128
#define NUM_DATAPOINTS_PERCHANNEL 1000000
#define NUM_INTERP_SAMPS 32
#define MAX_NUM_SPIKES  202000
const char FILENAME [] = "1000000DP_128CH_20WS_15CL";
//#define DEBUG_DATASET
//#define DEBUG_DATASET_ONLY
//#define DEBUG_SPLINE
//#define DEBUG_SPIKE_DETECT

typedef struct {
    double xip1;
    double xi;
    double zip1;
    double zi;
} spline_quad;

typedef struct {
    int dp;
    int cluster;
} spike_info;

typedef struct {
    int num_spikes;
    int num_clusters;
    int spk_cls_size[MAX_NUM_CLUSTER];
    //[cluster] [channel] [time point]
    float spk_cls_mean[MAX_NUM_CLUSTER][NUM_CHANNELS][NUM_INTERP_SAMPS];
    //int* cluster_variances;
} cls_info;

typedef struct {
    int clusterId;
    double dist;
} cluster_res;
