#ifndef TACO_TENSOR_T_H
#define TACO_TENSOR_T_H
#include <stdint.h>
typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;

struct taco_tensor_t {
  int32_t      order;         // tensor order (number of modes)
  int32_t*     dimensions;    // tensor dimensions
  int32_t      csize;         // component size
  int32_t*     mode_ordering; // mode storage ordering
  taco_mode_t* mode_types;    // mode storage types
  int32_t***   indices;       // tensor index data (per mode)
  float*     vals;          // tensor values
  int32_t      vals_size;     // values array size
};
#endif
