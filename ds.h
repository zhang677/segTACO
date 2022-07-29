#ifndef DS_H
#define DS_H

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <climits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

//#include <omp.h>

#include "mmio.h"
//#include "splatt.h"
#include <Eigen/Sparse>
#include "taco_tensor_t.h"

#define restrict __restrict__

#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))

typedef Eigen::SparseMatrix<float,Eigen::RowMajor,int> EigenCSR;
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> EigenRowMajor;
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> EigenColMajor;
typedef Eigen::Matrix<float,Eigen::Dynamic,1> EigenVector;


struct COO {
  int     m, n, nnz;
  int*    rows = nullptr;
  int*    cols = nullptr;
  float* vals = nullptr;
};

struct CSR {
  int     m, n;
  int*    pos  = nullptr;
  int*    crd  = nullptr;
  float* vals = nullptr;

  CSR() = default;
  CSR(int m, int n) : m(m), n(n) {}
};


/********** Utility Functions **********/


void free_matrix(COO matrix) {
  free(matrix.rows);
  free(matrix.cols);
  free(matrix.vals);
}

void free_matrix(CSR matrix) {
  free(matrix.pos);
  free(matrix.crd);
  free(matrix.vals);
}

COO read_matrix(const std::string matrix_path) {
  COO coo;

  std::fstream stream;
  stream.open(matrix_path, std::fstream::in);
  if (!stream) {
    stream.close();
    return coo;
  }

  std::string line;
  std::getline(stream, line);

  // Read Header
  std::stringstream lineStream(line);
  std::string head, type, formats, field, symmetry;
  lineStream >> head >> type >> formats >> field >> symmetry;
  assert(head=="%%MatrixMarket");
  // type = [matrix tensor]
  // formats = [coordinate array]
  assert((type == "matrix") || (type == "tensor"));
  // field = [real integer complex pattern]
  if (field != "real") {
    stream.close();
    return coo;
  }

  // symmetry = [general symmetric skew-symmetric Hermitian]
  if ((symmetry != "general") && (symmetry != "symmetric") && 
      (symmetry != "skew-symmetric")) {
    stream.close();
    return coo;
  }

  const bool symm = ((symmetry == "symmetric") || 
                     (symmetry == "skew-symmetric"));
  const bool skew = (symmetry == "skew-symmetric");

  std::getline(stream, line);

  // Skip comments at the top of the file
  std::string token;
  do {
    std::stringstream lineStream(line);
    lineStream >> token;
    if (token[0] != '%') {
      break;
    }
  } while (std::getline(stream, line));

  // The first non-comment line is the header with dimensions
  std::vector<size_t> dimensions;
  char* linePtr = (char*)line.data();
  while (size_t dimension = std::strtoull(linePtr, &linePtr, 10)) {
    dimensions.push_back(dimension);
  }

  assert(dimensions.size() == 3);
  size_t nnz = dimensions[2];
  
  coo.rows = (int*)malloc(sizeof(int) * nnz * (1 + symm));
  coo.cols = (int*)malloc(sizeof(int) * nnz * (1 + symm));
  coo.vals = (float*)malloc(sizeof(float) * nnz * (1 + symm));

  std::vector<int> coordinates(2);
  for (nnz = 0; std::getline(stream, line) && nnz <= INT_MAX; nnz++) {
    linePtr = (char*)line.data();
    for (size_t i = 0; i < dimensions.size(); i++) {
      const size_t index = strtoull(linePtr, &linePtr, 10);
      coordinates[i] = static_cast<int>(index);
    }

    const int i = coordinates[0] - 1;
    const int j = coordinates[1] - 1;
    float val = strtod(linePtr, &linePtr);

    coo.rows[nnz] = i;
    coo.cols[nnz] = j;
    coo.vals[nnz] = val;

    if (symm && i != j) {
      nnz++;

      if (skew) {
        val = -1.0 * val;
      }

      coo.rows[nnz] = j;
      coo.cols[nnz] = i;
      coo.vals[nnz] = val;
    }
  }

  stream.close();

  if (nnz > INT_MAX) {
    free_matrix(coo);
    coo.rows = nullptr;
    coo.cols = nullptr;
    coo.vals = nullptr;
    return coo;
  }

  coo.m = static_cast<int>(dimensions[0]);
  coo.n = static_cast<int>(dimensions[1]);
  coo.nnz = nnz;

  return coo;
}

// Load sparse matrix from an mtx file. Only non-zero positions are loaded,
// and values are dropped.
void read_mtx_file(const std::string filename, int &nrow, int &ncol, int &nnz,
                   std::vector<int> &csr_indptr_buffer,
                   std::vector<int> &csr_indices_buffer,
                   std::vector<int> &coo_rowind_buffer) {
  FILE *f;

  if ((f = fopen(filename.c_str(), "r")) == NULL) {
    printf("File %s not found", filename.c_str());
    exit(EXIT_FAILURE);
  }

  MM_typecode matcode;
  // Read MTX banner
  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process this file.\n");
    exit(EXIT_FAILURE);
  }
  if (mm_read_mtx_crd_size(f, &nrow, &ncol, &nnz) != 0) {
    printf("Could not process this file.\n");
    exit(EXIT_FAILURE);
  }
  // printf("Reading matrix %d rows, %d columns, %d nnz.\n", nrow, ncol, nnz);

  /// read tuples

  std::vector<std::tuple<int, int>> coords;
  int row_id, col_id;
  float dummy;
  for (int64_t i = 0; i < nnz; i++) {
    if (fscanf(f, "%d", &row_id) == EOF) {
      std::cout << "Error: not enough rows in mtx file.\n";
      exit(EXIT_FAILURE);
    } else {
      fscanf(f, "%d", &col_id);
      if (mm_is_integer(matcode) || mm_is_real(matcode)) {
        fscanf(f, "%f", &dummy);
      } else if (mm_is_complex(matcode)) {
        fscanf(f, "%f", &dummy);
        fscanf(f, "%f", &dummy);
      }
      // mtx format is 1-based
      coords.push_back(std::make_tuple(row_id - 1, col_id - 1));
    }
  }

  /// make symmetric

  if (mm_is_symmetric(matcode)) {
    std::vector<std::tuple<int, int>> new_coords;
    for (auto iter = coords.begin(); iter != coords.end(); iter++) {
      int i = std::get<0>(*iter);
      int j = std::get<1>(*iter);
      if (i != j) {
        new_coords.push_back(std::make_tuple(i, j));
        new_coords.push_back(std::make_tuple(j, i));
      } else
        new_coords.push_back(std::make_tuple(i, j));
    }
    std::sort(new_coords.begin(), new_coords.end());
    coords.clear();
    for (auto iter = new_coords.begin(); iter != new_coords.end(); iter++) {
      if ((iter + 1) == new_coords.end() || (*iter != *(iter + 1))) {
        coords.push_back(*iter);
      }
    }
  } else {
    std::sort(coords.begin(), coords.end());
  }
  /// generate csr from coo

  csr_indptr_buffer.clear();
  csr_indices_buffer.clear();

  int curr_pos = 0;
  csr_indptr_buffer.push_back(0);
  for (int64_t row = 0; row < nrow; row++) {
    while ((curr_pos < nnz) && (std::get<0>(coords[curr_pos]) == row)) {
      csr_indices_buffer.push_back(std::get<1>(coords[curr_pos]));
      coo_rowind_buffer.push_back(std::get<0>(coords[curr_pos]));
      curr_pos++;
    }
    // assert((std::get<0>(coords[curr_pos]) > row || curr_pos == nnz));
    csr_indptr_buffer.push_back(curr_pos);
  }
  nnz = csr_indices_buffer.size();
}

void fill_random(float array[], int size) {
  for (int i = 0; i < size; i++) {
    array[i] = (float)(std::rand() % 3) / 10;
  }
}

void fill_one(float array[], int size) {
  for (int i = 0; i < size; i++) {
    array[i] = 1.0f;
  }  
}

EigenRowMajor gen_row_major_matrix(int rows, int cols) {
  return EigenRowMajor::Ones(rows, cols);
  //return EigenRowMajor::Random(rows, cols) + 200 * EigenRowMajor::Ones(rows, cols);
}


COO new_get_matrix(const std::string filename) {
  COO coo;
  int nrow;
  int ncol;
  int nnz;
  std::vector<int> csr_indptr_buffer;
  std::vector<int> csr_indices_buffer;
  std::vector<int> coo_rowind_buffer;
  read_mtx_file(filename, nrow, ncol, nnz, csr_indptr_buffer, csr_indices_buffer, coo_rowind_buffer);
  
  coo.vals = (float *)malloc(sizeof(float) * nnz);
  fill_random(coo.vals, nnz);
  //fill_one(coo.vals, nnz);
  coo.m = nrow;
  coo.n = ncol;
  coo.nnz = nnz;
  coo.rows = (int *)malloc(sizeof(int) * nnz);
  coo.cols = (int *)malloc(sizeof(int) * nnz);
  
  for (int i=0 ; i<nnz ; i++){
    coo.cols[i] = csr_indices_buffer[i];
    coo.rows[i] = coo_rowind_buffer[i];
  }
  return coo;

}

void print_coo_matrix(const COO coo, const std::string log_path) {
  std::ofstream log_file;
  log_file.open(log_path, std::ofstream::out);
  log_file << "row: " << std::endl;
  int i=0;
  for (i=0;i<coo.nnz;i++) {
    if (i==coo.nnz-1) {
      log_file<<coo.rows[i]<<std::endl;
    }
    else {
      log_file<<coo.rows[i]<<",";
    }
  }
  log_file << "col: " << std::endl;
  for (i=0;i<coo.nnz;i++) {
    if (i==coo.nnz-1) {
      log_file<<coo.cols[i]<<std::endl;
    }
    else {
      log_file<<coo.cols[i]<<",";
    }
  }
  log_file << "vals: " << std::endl;
  for (i=0;i<coo.nnz;i++) {
    if (i==coo.nnz-1) {
      log_file<<coo.vals[i]<<std::endl;
    }
    else {
      log_file<<coo.vals[i]<<",";
    }
  }  
}

/*
splatt_csf* read_tensor(const std::string tensor_path) {
  double *cpd_opts = splatt_default_opts();
  cpd_opts[SPLATT_OPTION_NTHREADS] = omp_get_num_threads();
  cpd_opts[SPLATT_OPTION_NITER] = 0;
  cpd_opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
  cpd_opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;
  cpd_opts[SPLATT_OPTION_VERBOSITY] = SPLATT_VERBOSITY_NONE;

  splatt_idx_t nmodes;
  splatt_csf *tt;
  splatt_csf_load(tensor_path.c_str(), &nmodes, &tt, cpd_opts);
  
  splatt_free_opts(cpd_opts);

  return tt;
}

taco_tensor_t to_taco_tensor(const splatt_csf* tensor) {
  taco_tensor_t csft;

  csft.dimensions = new int32_t[tensor->nmodes];
  for (int i = 0; i < tensor->nmodes; ++i) {
    csft.dimensions[i] = tensor->dims[tensor->dim_perm[i]];
  }
  csft.indices = new int32_t**[tensor->nmodes];
  for (int i = 1; i < tensor->nmodes; ++i) {
    csft.indices[i] = new int32_t*[2];
    csft.indices[i][0] = (int32_t*)tensor->pt->fptr[i - 1];
    csft.indices[i][1] = (int32_t*)tensor->pt->fids[i];
  }
  csft.vals = (float*)tensor->pt->vals;

  return csft;
}
*/
// compress top level
void compress_top_level(taco_tensor_t &t) {
  int *pos_array = (int32_t*)malloc(sizeof(int32_t) * 2);
  pos_array[0] = 0; pos_array[1] = t.dimensions[0];
  int *crd_array = (int32_t*)malloc(sizeof(int32_t) * t.dimensions[0]);
  for (int i = 0; i < t.dimensions[0]; i++) {
    crd_array[i] = i;
  }
  t.indices[0] = new int32_t*[2];
  t.indices[0][0] = (int32_t *) pos_array;
  t.indices[0][1] = (int32_t *) crd_array;
}

taco_tensor_t to_taco_tensor(const COO matrix) {
  int32_t *pos = (int32_t*)malloc(sizeof(int32_t) * 2);
  pos[0] = 0; pos[1] = matrix.nnz;

  taco_tensor_t coot;

  coot.dimensions = new int32_t[2];
  coot.dimensions[0] = matrix.m;
  coot.dimensions[1] = matrix.n;
  coot.indices = new int32_t**[2];
  coot.indices[0] = new int32_t*[2];
  coot.indices[1] = new int32_t*[2];
  coot.indices[0][0] = (int32_t*)pos;
  coot.indices[0][1] = (int32_t*)matrix.rows;
  coot.indices[1][1] = (int32_t*)matrix.cols;
  //coot.vals = (int32_t*)matrix.vals;
  coot.vals = (float*)matrix.vals;
  return coot;
}

EigenCSR to_eigen_csr(const COO matrix, int shift = 0) {
  EigenCSR dst(matrix.m, matrix.n);
  std::vector<Eigen::Triplet<float>> tripletList;
  tripletList.reserve(matrix.nnz);
  if (shift == 0) {
    for (size_t i = 0; i < matrix.nnz; i++) {
      tripletList.push_back({matrix.rows[i], matrix.cols[i], matrix.vals[i]});
    }
  } else {
    for (size_t i = 0; i < matrix.nnz; i++) {
      const int col = (matrix.cols[i] + shift) % matrix.n;
      tripletList.push_back({matrix.rows[i], col, matrix.vals[i]});
    }
  }
  dst.setFromTriplets(tripletList.begin(), tripletList.end());
  dst.makeCompressed();
  return dst;
}

taco_tensor_t get_csr_taco_tensor(int rows, int cols) {
  taco_tensor_t csrt;

  csrt.dimensions = new int32_t[2];
  csrt.dimensions[0] = rows;
  csrt.dimensions[1] = cols;
  csrt.indices = new int32_t**[2];
  csrt.indices[1] = new int32_t*[2];

  return csrt;
}

taco_tensor_t to_taco_tensor(const EigenCSR& matrix) {
  taco_tensor_t csrt = get_csr_taco_tensor(matrix.rows(), matrix.cols());

  csrt.indices[1][0] = (int32_t*)matrix.outerIndexPtr();
  csrt.indices[1][1] = (int32_t*)matrix.innerIndexPtr();
  //csrt.vals = (int32_t*)matrix.valuePtr();
  csrt.vals = (float*)matrix.valuePtr();
  return csrt;
}

CSR get_csr_arrays(const taco_tensor_t matrix) {
  CSR csr;

  csr.m = matrix.dimensions[0];
  csr.n = matrix.dimensions[1];
  csr.pos = (int*)matrix.indices[1][0];
  csr.crd = (int*)matrix.indices[1][1];
  csr.vals = (float*)matrix.vals;

  return csr;
}
taco_tensor_t to_taco_tensor(const EigenRowMajor& matrix) {
  taco_tensor_t mt;

  mt.dimensions = new int32_t[2];
  mt.dimensions[0] = matrix.rows();
  mt.dimensions[1] = matrix.cols();
  //mt.vals = (int32_t*)matrix.data();
  mt.vals = (float*)matrix.data();
  int32_t order[2] = {0,1};
  mt.mode_ordering = order;

  return mt;
}

taco_tensor_t to_taco_tensor(const EigenColMajor& matrix) {
  taco_tensor_t vt;

  vt.dimensions = new int32_t[2];
  vt.dimensions[0] = matrix.rows();
  vt.dimensions[1] = matrix.cols();
  //vt.vals = (int32_t*)matrix.data();
  vt.vals = (float*)matrix.data();
  int32_t order[2] = {1,0};
  vt.mode_ordering = order;

  return vt;
}
/*
EigenColMajor gen_col_ones(int rows, int cols) {
  return Eigen::Matrix<float, 2, 3, Eigen::ColMajor>({1,2,3,4,5,6});
}

EigenColMajor gen_row_ones(int rows, int cols) {
  return Eigen::Matrix<float, 2, 3, Eigen::RowMajor>({1,2,3,4,5,6});
}
*/


EigenColMajor gen_col_major_matrix(int rows, int cols) {
  return EigenColMajor::Random(rows, cols) + EigenColMajor::Ones(rows, cols);
}
float compare_array(const float *x, const float *y, const size_t N) {
  float ret = 0.0;
  for (int i = 0; i < N; ++i) {
    //printf("x: %f, y: %f\n",x[i],y[i]);
    if (x[i] != 0.0) {
      const float diff = std::abs(y[i] / x[i] - 1.0);
      if (diff > ret) {
        ret = diff;
      }
    } else if (y[i] != 0.0) {
      return std::numeric_limits<float>::infinity();
    }
  }
  return ret;
}
float compare_array_my(const float *x, const float *y, const size_t N){
  float thres = 0.001;
  float count = 0.0;
  float col = 0;
  for (int i=0;i<N;i++){
    col = i%128;
    if(abs(x[i]-y[i])>thres){
       printf("col: %.0f, x: %.3f, y: %.3f\n",col,x[i],y[i]);
       count ++ ;
    }
    if(count>10000)
      break;
  }
  return count;
}
float compare_matrices(const taco_tensor_t a, const taco_tensor_t b) {
  assert(a.dimensions[0] == b.dimensions[0]);
  assert(a.dimensions[1] == b.dimensions[1]);
  const size_t N = a.dimensions[0] * a.dimensions[1];
  //return compare_array((float*)a.vals, (float*)b.vals, N);
  return compare_array_my((float*)a.vals, (float*)b.vals, N);
}
float compare_csr_val(const taco_tensor_t a, const taco_tensor_t b) {
  int B1_dimension = (int)(b.dimensions[0]);
  int* restrict B2_pos = (int*)(b.indices[1][0]);


  int upper = B2_pos[B1_dimension];


  float* restrict A_vals = (float*)(a.vals);
  float* restrict B_vals = (float*)(b.vals);
	
	float ret = 0.0;
	for (int i = 0; i < upper; ++i) {
		if(A_vals[i] != 0.0) {
			const float diff = std::abs(B_vals[i] / A_vals[i] - 1.0);
			if (diff > ret) {
				ret = diff;
			}
		} else if (B_vals[i] != 0.0) {
			return std::numeric_limits<float>::infinity();
		}
	}
	return ret;

}  
EigenVector gen_vector(int size) {
  return EigenVector::Random(size) + 2.0 * EigenVector::Ones(size);
}
taco_tensor_t to_taco_tensor(const EigenVector& vector) {
  taco_tensor_t vt;

  vt.dimensions = new int32_t[1];
  vt.dimensions[0] = vector.innerSize();
  //vt.vals = (int32_t*)vector.data();
  vt.vals = (float*)vector.data();
  return vt;
}
float compare_vectors(const taco_tensor_t a, const taco_tensor_t b) {
  assert(a.dimensions[0] == b.dimensions[0]);
  return compare_array((float*)a.vals, (float*)b.vals, a.dimensions[0]);
}

void print_matrix(const taco_tensor_t a) {
    const size_t N = a.dimensions[0] * a.dimensions[1];
    for (int i=0;i<N;i++) {
      printf("x[%d]: %.3f\n",i,a.vals[i]);
    }
}

float compare_array_int(const int32_t *x, const int32_t *y, const size_t N) {
  float ret = 0.0;
  for (int i = 0; i < N; ++i) {
    if(x[i] != y[i]){
      printf("x: %d, y: %d\n",x[i],y[i]);
      ret = ret + 1.0;
      if(ret>10000){
        break;
      }
    }
  }
  return ret;
}


double compare_array_float(const float *x, const float *y, const size_t N) {
  double ret = 0.0;
  for (int i = 0; i < N; ++i) {
    if (x[i] != 0.0) {
      const double diff = std::abs(y[i] / x[i] - 1.0);
      if (diff > ret) {
        ret = diff;
      }
    } else if (y[i] != 0.0) {
      return std::numeric_limits<float>::infinity();
    }
  }
  return ret;
}

double compare_matrices_float(const taco_tensor_t a, const taco_tensor_t b) {
  assert(a.dimensions[0] == b.dimensions[0]);
  assert(a.dimensions[1] == b.dimensions[1]);
  const size_t N = a.dimensions[0] * a.dimensions[1];
  return compare_array_my((float*)a.vals, (float*)b.vals, N);
}




/*
float compare_matrices_int(const taco_tensor_t a, const taco_tensor_t b) {
  assert(a.dimensions[0] == b.dimensions[0]);
  assert(a.dimensions[1] == b.dimensions[1]);
  const size_t N = a.dimensions[0] * a.dimensions[1];
  return compare_array_int(a.vals,b.vals, N);
}
*/



#endif