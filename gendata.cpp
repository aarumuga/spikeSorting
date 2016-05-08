#include <cstdlib>
#include <fstream>
#include <iostream>

#define WINDOW_SIZE 20
#define NUM_CHANNELS 128
#define DATA_LENGTH 1000000
#define NOISE_DATA_RATIO 3
#define NUM_CLUSTERS 15

bool include_noise = true;
int main() {

  const int dp_per_ch = DATA_LENGTH+(NOISE_DATA_RATIO+1)*WINDOW_SIZE;
  double* data = new double[dp_per_ch * NUM_CHANNELS];
  double cldata[NUM_CLUSTERS][NUM_CHANNELS][WINDOW_SIZE];
  int cluster_count[NUM_CLUSTERS];

  /* populate cluster data */

  for (int i = 0; i < NUM_CLUSTERS; i++) {
      cluster_count[i] = 0;
      for(int k = 0; k < NUM_CHANNELS; k++) {
          cldata[i][k][0] = 0.5f;
          cldata[i][k][1] = 0.6f;
          cldata[i][k][2] = 0.7f;
          cldata[i][k][3] = 0.8f;
          cldata[i][k][4] = 0.9f;
          cldata[i][k][5] = 0.5f;

          for (int j = 6; j < WINDOW_SIZE; j++) {
              //cldata[i][k][j] = ((double)rand())/((double)RAND_MAX*2) + 0.5f;
              cldata[i][k][j] = ((double)rand())/((double)RAND_MAX*2);
          }
      }
  }

  //generate datasets.

  int iter = 0;
  while (iter < DATA_LENGTH) {
      // Add noise before every cluster
      if (include_noise) {
          int lnoise = rand() % (NOISE_DATA_RATIO*WINDOW_SIZE);
          for (int i = 0; i < lnoise; i++) {
              //add noise to each channel
              for (int j = 0; j < NUM_CHANNELS; j++) {
                  data[iter+i+j*dp_per_ch] = ((double)rand())/((double)RAND_MAX*2);
              }
          }
          iter += lnoise;
      }

      // add spike from random cluster.
      int sp = rand() % NUM_CLUSTERS;
      cluster_count[sp]++;
      for (int i = 0; i < NUM_CHANNELS; i++) {
          for (int j = 0; j < WINDOW_SIZE; j++) {
              data[iter+j+(i*dp_per_ch)] = cldata[sp][i][j];
          }
      }
      iter += WINDOW_SIZE;
  }


  char file_name[200];
  sprintf(file_name, "%dDP_%dCH_%dWS_%dCL", DATA_LENGTH, NUM_CHANNELS, WINDOW_SIZE, NUM_CLUSTERS);
  std::ofstream df;
  df.open(file_name, std::ofstream::out | std::ofstream::binary);
  for (int i = 0; i < NUM_CHANNELS; i++) {
    df.write((char*)(&(data[i*dp_per_ch])), DATA_LENGTH*sizeof(double));
  }
  df.close();
  sprintf(file_name, "%dDP_%dCH_%dWS_%dCL_result", DATA_LENGTH, NUM_CHANNELS, WINDOW_SIZE, NUM_CLUSTERS);

  df.open(file_name, std::ofstream::out);
  df << " WINDOWSIZE : " << WINDOW_SIZE << std::endl;
  df << " NOISEDATARATIO : " << NOISE_DATA_RATIO << std::endl;
  df << " NUMDATAPOINTS PER CHANNEL : " << DATA_LENGTH << std::endl;
  df << " NUM_CHANNELS :" << NUM_CHANNELS << std::endl;
  df << " NUM_CLUSTERs : " << NUM_CLUSTERS << std::endl;
  int total = 0;
  for (int i = 0; i < NUM_CLUSTERS; i++) {
      df << " CLUSTER " << i << "  count :" << cluster_count[i] << std::endl;
      total += cluster_count[i];
  }
  df << "total spikes : " << total << std::endl;

  for (int i = 0; i < NUM_CLUSTERS; i++) {
      df << "cluster : " << i <<  " datapoints " << std::endl;
      for (int j = 0; j < NUM_CHANNELS; j++) {
          df << "channel : " << j << std::endl;
          for (int k = 0; k < WINDOW_SIZE; k++) {
              df << cldata[i][j][k] << ", ";
          }
          df << std::endl;
      }
      df << std::endl;
  }

  df << "initial data points of all channels for 2 windowsize" << std::endl;
  int temp = 0;
  while (temp < 2) {
      for (int i = 0; i < NUM_CHANNELS; i++) {
          df << std::endl;
          df << "CHANNEL " << i << " datapoints " << std::endl;
          for (int j = 0; j < WINDOW_SIZE; j++) {
              df << data[(i*dp_per_ch) + j + (temp * WINDOW_SIZE)] << " , ";
          }
      }
      temp++;
  }
  df.close();

  delete data;
}
