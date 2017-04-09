
# blas_order_type
const blas_rowmajor = 101
const blas_colmajor = 102

# blas_trans_type
const blas_no_trans   = 111
const blas_trans      = 112
const blas_conj_trans = 113

# blas_uplo_type
const blas_upper = 121
const blas_lower = 122

# blas_diag_type
const blas_non_unit_diag = 131
const blas_unit_diag     = 132

# blas_side_type
const blas_left_side  = 141
const blas_right_side = 142

# blas_base_type
const blas_zero_base = 221
const blas_one_base  = 222

# blas_symmetry_type
const blas_general          = 231
const blas_symmetric        = 232
const blas_hermitian        = 233
const blas_triangular       = 234
const blas_lower_triangular = 235
const blas_upper_triangular = 236
const blas_lower_symmetric  = 237
const blas_upper_symmetric  = 238
const blas_lower_hermitian  = 239
const blas_upper_hermitian  = 240

# blas_field_type
const blas_complex          = 241
const blas_real             = 242
const blas_double_precision = 243
const blas_single_precision = 244

# blas_size_type
const blas_num_rows     = 251
const blas_num_cols     = 252
const blas_num_nonzeros = 253

# blas_handle_type
const blas_invalid_handle = 261
const blas_new_handle     = 262
const blas_open_handle    = 263
const blas_valid_handle   = 264

# blas_sparsity_optimization_type
const blas_regular     = 271
const blas_irregular   = 272
const blas_block       = 273
const blas_unassembled = 274

# blas_rsb_ext_type
const blas_rsb_spmv_autotuning_on      = 6660
const blas_rsb_spmv_autotuning_off     = 6661
const blas_rsb_spmv_n_autotuning_on    = 6662
const blas_rsb_spmv_n_autotuning_off   = 6663
const blas_rsb_spmv_t_autotuning_on    = 6664
const blas_rsb_spmv_t_autotuning_off   = 6665
const blas_rsb_autotune_next_operation = 6666
const blas_rsb_rep_rsb                 = 9995
const blas_rsb_rep_csr                 = 9996
const blas_rsb_rep_coo                 = 9997
const blas_rsb_duplicates_ovw          = 9998
const blas_rsb_duplicates_sum          = 9999

const RSB_FLAG_FORTRAN_INDICES_INTERFACE = 0x000001
const RSB_FLAG_C_INDICES_INTERFACE       = 0x000000

