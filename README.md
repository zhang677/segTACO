# segTACO

# Get Started

Change the `MATRICES_DIR` in `run_gpu.sh` to the path to sparse matrices.

```
make clean
make
./run_gpu.sh
```

`new_names.txt` contains all the sparse matrices we use. It has 740 matrices in total. It is less than 956 as stated in [DA-SpMMul](https://arxiv.org/pdf/2202.08556.pdf). But we have confirmed with the authors about the number. 