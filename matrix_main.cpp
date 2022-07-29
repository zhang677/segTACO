#include "ds.h"
#include "timers.h"
#include "matrix_experiments/spmm_csr_gpu.h"
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>

Timer timer;
std::ofstream log_file;

int main(int argc, char* argv[]) {

  const std::string matrix_dir  = (argc > 1) ? argv[1] : "/data/scratch/changwan/florida_all/2cubes_sphere/";
  const std::string matrix_name = (argc > 2) ? argv[2] : "2cubes_sphere";
  const std::string experiment  = (argc > 3) ? argv[3] : "spmv_csr_gpu";
  const std::string results_dir = (argc > 4) ? argv[4] : ".";
  const std::string hardware       = (argc > 5) ? argv[5] : "bench";


  //const bool do_verify = (mode == "validate");
  //const bool do_verify = true;
  const bool do_verify = false;
  const int num_cols = 4;
  const std::string matrix_path = matrix_dir + "/" + matrix_name + ".mtx";
  //const COO input_matrix = read_matrix(matrix_path);
  const COO input_matrix = new_get_matrix(matrix_path);
  //std::cout<<input_matrix.vals<<std::endl;
  if (input_matrix.vals) {
    const std::string log_path = results_dir + "/" + experiment + "_" + std::to_string(num_cols) + "_" + hardware + ".csv";
    log_file.open(log_path, std::ofstream::app);
    
    if(log_file){
      std::cout<<log_path+" has been opened"<<std::endl;
    }

    if (experiment == "spmv_csr_gpu") {
      ;//spmv_csr_gpu(input_matrix, matrix_name, do_verify);
    }
    else if (experiment == "spmm_csr_gpu") {
      spmm_csr_gpu(input_matrix, matrix_name, do_verify, num_cols);
    }
    else if (experiment == "sddmm_csr_gpu") {
      //sddmm_csr_gpu(input_matrix, matrix_name, do_verify);
    }
    else if (experiment == "spmspv_csr_gpu") {
      ;//spmspv_csr_gpu(input_matrix, matrix_name, do_verify);
    }


    free_matrix(input_matrix);
    log_file.close();
  }

  return 0;
}
