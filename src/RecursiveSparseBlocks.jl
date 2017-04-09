
module RecursiveSparseBlocks

export SparseMatrixRSB

include("../deps/deps.jl")
include("constants.jl")


typealias BLASSparseMatrix Cint
typealias RSBErr Cint


function __init__()
    # init library with default options
    err = ccall((:rsb_lib_init, librsb), RSBErr,
                (Ptr{Void},), C_NULL)

    if err != 0
        error("rsb_lib_init failed with code $(err)")
    end
end


type SparseMatrixRSB{T} <: AbstractSparseMatrix{T, Cint}
    ptr::BLASSparseMatrix
    transpose::Bool
    conj::Bool
end


function destroy(M::SparseMatrixRSB)
    err = ccall((:BLAS_usds, librsb), Cint, (BLASSparseMatrix,), M.ptr)
    if err != 0
        error("BLAS_usds call failed with code $(err)")
    end
end


for ((uscrbegin, uscrins, uscrend), T) in ((("BLAS_suscr_begin", "BLAS_suscr_insert_col", "BLAS_suscr_end"), :Float32),
                                           (("BLAS_duscr_begin", "BLAS_duscr_insert_col", "BLAS_duscr_end"), :Float64),
                                           (("BLAS_cuscr_begin", "BLAS_cuscr_insert_col", "BLAS_cuscr_end"), :Complex64),
                                           (("BLAS_zuscr_begin", "BLAS_zuscr_insert_col", "BLAS_zuscr_end"), :Complex128))
    @eval begin
        function SparseMatrixRSB(A::SparseMatrixCSC{$T})
            m, n = size(A)
            nnz = length(A.nzval)

            ptr = ccall(($uscrbegin, librsb), BLASSparseMatrix, (Cint, Cint), m, n)
            if ptr == -1
                error("failed to initialize matrix")
            end

            colrowval = Cint[]

            for j in 1:n
                colnnz = A.colptr[j+1] - A.colptr[j]
                if length(colrowval) < colnnz
                    resize!(colrowval, colnnz)
                end
                copy!(colrowval, 1, A.rowval, A.colptr[j], colnnz)
                colrowval .-= 1

                err = ccall(($uscrins, librsb), Cint,
                            (BLASSparseMatrix, Cint, Cint, Ptr{$T}, Ptr{Cint}),
                            ptr, j - 1, colnnz, pointer(A.nzval, A.colptr[j]),
                            colrowval)

                if err != 0
                    error("insert_col failed with code $(err)")
                end
            end


            err = ccall(($uscrend, librsb), Cint, (BLASSparseMatrix,), ptr)
            if err != 0
                error("BLAS_suscr_end failed with code $(err)")
            end

            ret = SparseMatrixRSB{$T}(ptr, false, false)
            finalizer(ret, destroy)
            return ret
        end
    end
end


function Base.convert{T}(::Type{SparseMatrixCSC}, A::SparseMatrixRSB{T})
    I, J, V = findnz(A)
    m, n = size(A)
    return sparse(I, J, V, m, n)
end


function Base.findnz{T}(A::SparseMatrixRSB{T})
    N = nnz(A)
    V = Array(T, N)
    I = Array(Cint, N)
    J = Array(Cint, N)

    mtxptr = ccall((:rsb_blas_get_mtx, librsb), Ptr{Void},
                   (BLASSparseMatrix,), A.ptr)
    if mtxptr == C_NULL
        error("rsb_blas_get_mtx did not return a valid matrix pointer")
    end

    err = ccall((:rsb_mtx_get_coo, librsb), RSBErr,
              (Ptr{Void}, Ptr{T}, Ptr{Cint}, Ptr{Cint}, Cint),
              mtxptr,     V,      I,         J,
              RSB_FLAG_FORTRAN_INDICES_INTERFACE)
    if err != 0
        error("rsb_mtx_get_coo failed with code $(err)")
    end

    return (I, J, V)
end


for ((uscrbegin, uscrins, uscrend), T) in ((("BLAS_suscr_begin", "BLAS_suscr_insert_entries", "BLAS_suscr_end"), :Float32),
                                           (("BLAS_duscr_begin", "BLAS_duscr_insert_entries", "BLAS_duscr_end"), :Float64),
                                           (("BLAS_cuscr_begin", "BLAS_cuscr_insert_entries", "BLAS_cuscr_end"), :Complex64),
                                           (("BLAS_zuscr_begin", "BLAS_zuscr_insert_entries", "BLAS_zuscr_end"), :Complex128))
    @eval begin
        function Base.copy(A::SparseMatrixRSB{$T})
            # Dump the matrix to COO, then insert those entries into a new matrix.
            # This seems a little silly, but I can't find another way to clone a
            # blas_sparse_matrix.

            I, J, V = findnz(A)
            m, n = size(A)

            ptr = ccall(($uscrbegin, librsb), BLASSparseMatrix, (Cint, Cint), m, n)
            if ptr == -1
                error("failed to initialize matrix")
            end

            err = ccall(($uscrins, librsb), Cint,
                        (BLASSparseMatrix, Cint, Ptr{$T}, Ptr{Cint}, Ptr{Cint}),
                        ptr, length(V), V, I, J)
            if err != 0
                error("insert_entries failed with code $(err)")
            end

            err = ccall(($uscrend, librsb), Cint, (BLASSparseMatrix,), ptr)

            if err != 0
                error("BLAS_suscr_end failed with code $(err)")
            end

            ret = SparseMatrixRSB{$T}(ptr, false, false)
            finalizer(ret, destroy)
            return ret
        end
    end
end


"""
Get matrix property.
"""
function blas_usgp(ptr::BLASSparseMatrix, pname::Integer)
    return ccall((:rsb_wp__BLAS_usgp, librsb), Cint, (BLASSparseMatrix, Cint),
                 ptr, pname)
end


"""
Set matrix property.
"""
function blas_ussp(ptr::BLASSparseMatrix, pname::Integer)
    return ccall((:rsb_wp__BLAS_ussp, librsb), Cint, (BLASSparseMatrix, Cint),
                 ptr, pname)
end


function Base.nnz(A::SparseMatrixRSB)
    return blas_usgp(A.ptr, blas_num_nonzeros)
end


function Base.size(A::SparseMatrixRSB)
    return (Int(blas_usgp(A.ptr, blas_num_rows)),
            Int(blas_usgp(A.ptr, blas_num_cols)))
end


function Base.size(A::SparseMatrixRSB, k::Integer)
    if k == 1
        return Int(blas_usgp(A.ptr, blas_num_rows))
    elseif k == 2
        return Int(blas_usgp(A.ptr, blas_num_cols))
    else
        return 1
    end
end

for (f, T) in (("BLAS_susget_element", Float32),
               ("BLAS_dusget_element", Float64),
               ("BLAS_cusget_element", Complex64),
               ("BLAS_zusget_element", Complex128))
    @eval begin
        function Base.getindex(A::SparseMatrixRSB{$T}, i::Integer, j::Integer)
            y = Ref{$T}()
            err = ccall(($f, librsb), Cint, (BLASSparseMatrix, Cint, Cint, Ptr{$T}),
                        A.ptr, i - 1, j - 1, y)
            if err != 0
                # Unfortunately I'm not sure there is a way to distinguish
                # between zero-entry and actual error
                return zero($T)
            end
            return y.x
        end
    end
end


for (f, T) in (("BLAS_susset_element", Float32),
               ("BLAS_dusset_element", Float64),
               ("BLAS_cusset_element", Complex64),
               ("BLAS_zusset_element", Complex128))
    @eval begin
        function Base.setindex!(A::SparseMatrixRSB{$T}, value_, i::Integer, j::Integer)
            value = $T(value_)
            y = Ref{$T}(value)
            err = ccall(($f, librsb), Cint, (BLASSparseMatrix, Cint, Cint, Ptr{$T}),
                        A.ptr, i - 1, j - 1, y)
            if err != 0
                error("Cannot set zero entry ($i, $j) of a SparseMatrixRSB")
            end
            return value
        end
    end

end


function Base.show(io::IO, ::MIME"text/plain", A::SparseMatrixRSB)
end


for (f, T) in (("BLAS_susget_diag", Float32),
               ("BLAS_dusget_diag", Float64),
               ("BLAS_cusget_diag", Complex64),
               ("BLAS_zusget_diag", Complex128))
    @eval begin
        function Base.diag(A::SparseMatrixRSB{$T})
            d = Array($T, min(size(A)...))
            err = ccall(($f, librsb), Cint, (BLASSparseMatrix, Ptr{$T}),
                        A.ptr, d)
            if err != 0
                error("usget_diag failed with code $(err)")
            end
            return d
        end
    end
end


function Base.full{T}(A::SparseMatrixRSB{T})
    B = zeros(T, size(A))
    I, J, V = findnz(A)
    for (i, j, v) in zip(I, J, V)
        B[i, j] = v
    end
    return B
end


for (f, T) in (("BLAS_susmv", Float32),
               ("BLAS_dusmv", Float64),
               ("BLAS_cusmv", Complex64),
               ("BLAS_zusmv", Complex128))
    # TODO: consider trans and conj state
    @eval begin
        function Base.A_mul_B!(y::Vector{$T}, A::SparseMatrixRSB{$T}, x::Vector{$T})
            fill!(y, zero($T))
            err = ccall(($f, librsb), Cint,
                        (Cint,             # transA
                         $T,               # alpha
                         BLASSparseMatrix, # A
                         Ptr{$T},          # x
                         Cint,             # incx
                         Ptr{$T},          # y
                         Cint),            # incy
                        blas_no_trans, one($T), A.ptr, x, 1, y, 1)
            if err != 0
                error("BLAS_susmv failed with code $(err)")
            end
            return y
        end
    end
end


# TODO:
# show
# convert{S, T}(::Type{SparseMatrixRSB{S}}, ::SparseMatrixRSB{T})
# transpose / transpose! (set flag)
# conj / conj! (set flag)
# SM*DV / A_mul_B! (BLAS_?usmv)
# SM*DM (BLAS_?usmm)
# A_ldiv_B! / trisolve (BLAS_?ussv, BLAS_?ussm)
# + (I think we just insert elements with the right flag set if we want to add)

end



