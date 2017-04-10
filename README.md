
# RecursiveSparseBlocks

Julia interface to [librsb](http://librsb.sourceforge.net/), a general purpose
sparse matrix library that claims to be broadly competitive with MKL on the
subset of functions it implements but is open source and more permissively
licenced (GPL3).

## Usage

`RecursiveSparseBlocks` exports a `SparseMatrixRSB` which works similarly to
`SparseMatrixCSC`, but with a smaller subset of features.

## Benchamrks

For large matrices `SparseMatrixRSB` is much faster that Julia's built-in
`SparseMatrixCSC` (in no small part because the former is multithreaded).

The following plots show elapsed time multiplying random sparse matrices by
random dense vectors, each with a density of 1e-4. The first shown `Float32`
performance, the second `Float64`. These benchmarks were run on an 8-core cpu.

![](https://github.com/dcjones/RecursiveSparseBlocks.jl/raw/master/bench-float32.png)

![](https://github.com/dcjones/RecursiveSparseBlocks.jl/raw/master/bench-float64.png)


