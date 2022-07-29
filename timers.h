#ifndef TACO_UTIL_BENCHMARK_H
#define TACO_UTIL_BENCHMARK_H

#include <cassert>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <cstdint>
#include <random>

#include <omp.h>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> TimePoint;



/// Monotonic timer that can be called multiple times and that computes
/// statistics such as mean and median from the calls.
class Timer {
public:
  Timer() {
    const auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    srand(seed);

    dummy = new double[dummy_size];
    for (int i = 0; i < dummy_size; i++) {
      dummy[i] = (double)rand() / RAND_MAX;
    }

    indices = new int[dummy_size];
    std::iota(indices, indices + dummy_size, 0);
    std::shuffle(indices, indices + dummy_size, std::default_random_engine(seed));
  }

  ~Timer() {
    delete[] dummy;
  }

  void start() {
    begin = std::chrono::high_resolution_clock::now();
  }

  void stop() {
    auto end = std::chrono::high_resolution_clock::now();
    if (dummy[0] < 0.0) {
      // should never get here
      std::cerr << "Unexpected error!" << std::endl;
      exit(1);
    }
    res = std::chrono::duration<double,std::milli>(end - begin).count();
  }

  double get_result() {
    return res;
  }

  void clear_cache() {
    #pragma omp parallel num_threads(4)
    {
      double ret = 0.0;
      for (int i = 0; i < dummy_size; i++) {
        //ret += dummy[rand() % dummy_size];
        //ret += dummy[dummy_size - i - 1];
        //ret += dummy[i];
        ret += dummy[indices[i]];
      }
      const int tid = omp_get_thread_num();
      dummy[tid] = ret;
    }
  }

private:
  TimePoint begin;
  double res = -1;

  const int dummy_size = 10000000;
  double* dummy;
  int* indices;
};

extern Timer timer;
extern std::ofstream log_file;


#include <cuda_runtime.h>
#include "gpu_kernels.cuh"

void log_result_gpu(const std::string& kernel, const std::string& platform,
                const std::string& format, const std::string& library,
                const std::string& tensor, const int trial) {
  //const auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
  //std::cout<<"logging to file"<<std::endl;
  log_file << kernel << "," << platform << "," << format << "," << library
           << "," << tensor << "," << get_gpu_timer_result() << std::endl;
}


void log_result(const std::string& kernel, const std::string& platform,
                const std::string& format, const std::string& library,
                const std::string& tensor, const int trial) {
  const auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
  log_file << kernel << "," << platform << "," << format << "," << library
           << "," << tensor << "," << trial << "," << timestamp 
           << "," << timer.get_result() << std::endl;
}

#define TIME_COLD(CODE) { \
    timer.clear_cache();  \
    timer.start();        \
    CODE;                 \
    timer.stop();         \
  }

#define TIME_WARM(CODE) { \
    timer.start();        \
    CODE;                 \
    timer.stop();         \
  }

#define RUN(CODE,TRIALS,EXPERIMENT,PLATFORM,FORMAT,LIB,TENSOR) {        \
    for (int trial = 0; trial < (TRIALS); trial++) {                    \
      CODE;                                                             \
      log_result(EXPERIMENT, PLATFORM, FORMAT, LIB, TENSOR, trial + 1); \
    }                                                                   \
  }

#define RUN_GPU(CODE,TRIALS,EXPERIMENT,PLATFORM,FORMAT,LIB,TENSOR) {        \
    for (int trial = 0; trial < (TRIALS); trial++) {                        \
      CODE;                                                                 \
      log_result_gpu(EXPERIMENT, PLATFORM, FORMAT, LIB, TENSOR, trial + 1); \
    }                                                                       \
  }

#endif
