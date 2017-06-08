
module RecursiveSparseBlocks

export SparseMatrixRSB

include("../deps/deps.jl")
include("constants.jl")


const RSBCooIdx = Cint
const RSBNnzIdx = Cint
const RSBType   = Cchar
const RSBFlags  = Cint
const RSBErr    = Cint
const RSBChar   = Cchar
const RSBBlkIdx = Cint
const RSBTrans  = RSBFlags


function __init__()
    # init library with default options
    err = ccall((:rsb_lib_init, librsb), RSBErr,
                (Ptr{Void},), C_NULL)

    if err != 0
        error("rsb_lib_init failed with code $(err)")
    end
end


type SparseMatrixRSB{T} <: AbstractSparseMatrix{T, Cint}
    ptr::Ptr{Void}
end


function rsb_mtx_free(A::SparseMatrixRSB)
    ccall((:rsb_mtx_free, librsb), Ptr{Void}, (Ptr{Void},), A.ptr)
end


function check_rsb_error(err::RSBErr)
    if err != RSB_ERR_NO_ERROR
        buflen = 5000
        buf = Array{UInt8}(buflen)
        ccall((:rsb_strerror_r, librsb), RSBErr, (RSBErr, Ptr{RSBChar}, Csize_t),
              err, buf, buflen)
        i = findfirst(0x0, buf)
        if i == 0
            error("SparseMatrixRSB operation failed with code $(err)")
        else
            error(String(buf[1:i-1]))
        end
    end
end


@inline function rsb_type_code{T}(::Type{T})
    typ = T == Float32    ? 'S' :
          T == Float64    ? 'D' :
          T == Complex64  ? 'C' :
          T == Complex128 ? 'Z' :
          error("librsb does not support matrices of type $T")
    return typ
end


function SparseMatrixRSB{T}(A::SparseMatrixCSC{T})
    typ = rsb_type_code(T)
    m, n = size(A)
    nnz = length(A.nzval)

    IA = Vector{RSBCooIdx}(A.rowval)
    CP = Vector{RSBCooIdx}(A.colptr)

    # make zero based
    IA .-= 1
    CP .-= 1

    err = Ref{RSBErr}()
    ptr = ccall((:rsb_mtx_alloc_from_csc_const, librsb), Ptr{Void},
                (Ptr{Void}, Ptr{RSBCooIdx}, Ptr{RSBCooIdx}, RSBNnzIdx, RSBType,
                 RSBCooIdx, RSBCooIdx, RSBBlkIdx, RSBBlkIdx, RSBFlags,
                 Ptr{RSBErr}),
                A.nzval, IA, CP, nnz, typ, m, n,
                RSB_DEFAULT_ROW_BLOCKING, RSB_DEFAULT_COL_BLOCKING,
                RSB_FLAG_DEFAULT_RSB_MATRIX_FLAGS, err)
    check_rsb_error(err.x)

    ret = SparseMatrixRSB{T}(ptr)
    finalizer(ret, rsb_mtx_free)
    return ret
end


function SparseMatrixRSB{T}(I_, J_, V::Vector{T}, m::Integer, n::Integer)
    typ = rsb_type_code(T)
    nnz = length(V)
    I = Vector{RSBCooIdx}(I_)
    J = Vector{RSBCooIdx}(J_)

    # make zero based
    I .-= 1
    J .-= 1

    err = Ref{RSBErr}()
    ptr = ccall((:rsb_mtx_alloc_from_coo_const, librsb), Ptr{Void},
                (Ptr{Void}, Ptr{RSBCooIdx}, Ptr{RSBCooIdx}, RSBNnzIdx, RSBType,
                 RSBCooIdx, RSBCooIdx, RSBBlkIdx, RSBBlkIdx, RSBFlags,
                 Ptr{RSBErr}),
                V, I, J, nnz, typ, m, n,
                RSB_DEFAULT_ROW_BLOCKING, RSB_DEFAULT_COL_BLOCKING,
                RSB_FLAG_DEFAULT_RSB_MATRIX_FLAGS, err)
    check_rsb_error(err.x)

    ret = SparseMatrixRSB{T}(ptr)
    finalizer(ret, rsb_mtx_free)
    return ret
end


function Base.convert{T}(::Type{SparseMatrixCSC}, A::SparseMatrixRSB{T})
    I, J, V = findnz(A)
    m, n = size(A)
    return sparse(I, J, V, m, n)
end


function Base.findnz{T}(A::SparseMatrixRSB{T})
    N = nnz(A)
    V = Array{T}(N)
    I = Array{RSBCooIdx}(N)
    J = Array{RSBCooIdx}(N)

    err = ccall((:rsb_mtx_get_coo, librsb), RSBErr,
                (Ptr{Void}, Ptr{Void}, Ptr{RSBCooIdx}, Ptr{RSBCooIdx},
                 RSBFlags),
                A.ptr, V, I, J, RSB_FLAG_FORTRAN_INDICES_INTERFACE)
    check_rsb_error(err)

    return (I, J, V)
end


function Base.copy{T}(A::SparseMatrixRSB{T})
    typ = rsb_type_code(T)
    Bptr = Ref{Ptr{Void}}()
    err = ccall((:rsb_mtx_clone, librsb), RSBErr,
                (Ptr{Ptr{Void}}, RSBType, RSBTrans, Ptr{Void}, Ptr{Void}, RSBFlags),
                BPtr, typ, RSB_TRANSPOSITION_N, C_NULL, A.ptr,
                RSB_FLAG_DEFAULT_RSB_MATRIX_FLAGS)
    check_rsb_error(err)


    ret = SparseMatrixRSB{T}(Bptr.x)
    finalizer(ret, rsb_mtx_free)
    return ret
end


function Base.eltype{T}(A::SparseMatrixRSB{T})
    return T
end


function Base.show(io::IO, ::MIME"text/plain", A::SparseMatrixRSB)
    m, n = size(A)
    print(io, m, "×", n, " sparse RSB matrix with ", nnz(A), " ", eltype(A),
          " stored entries")
    if nnz(A) != 0
        print(io, ":")
        show(io, A)
    end
end


Base.show(io::IO, S::SparseMatrixRSB) = Base.show(convert(IOContext, io),
                                                  S::SparseMatrixRSB)

function Base.show(io::IOContext, S::SparseMatrixRSB)
    # Taken mostly from SparseMatrixCSC show function

    if nnz(S) == 0
        return show(io, MIME("text/plain"), S)
    end

    limit::Bool = get(io, :limit, false)
    if limit
        rows = displaysize(io)[1]
        half_screen_rows = div(rows - 8, 2)
    else
        half_screen_rows = typemax(Int)
    end
    pad = ndigits(max(size(S)...))
    k = 0
    sep = "\n  "
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end

    I, J, V = findnz(S)
    k = 0
    for (i, j, v) in zip(I, J, V)
        if k < half_screen_rows || k > nnz(S)-half_screen_rows
            print(io, sep, '[',
                  rpad(i, pad), ", ",
                  lpad(j, pad), "]  =  ")
            Base.show(io, v)
        elseif k == half_screen_rows
            print(io, sep, '\u22ee')
        end
        k += 1
    end
end


function Base.nnz(A::SparseMatrixRSB)
    out = Ref{RSBNnzIdx}()
    err = ccall((:rsb_mtx_get_info, librsb), RSBErr,
                (Ptr{Void}, Cint, Ptr{Void}),
                A.ptr, RSB_MIF_MATRIX_NNZ__TO__RSB_NNZ_INDEX_T, out)
    check_rsb_error(err)
    return Int(out.x)
end


function rsb_get_nrows(A::SparseMatrixRSB)
    out = Ref{RSBCooIdx}()
    err = ccall((:rsb_mtx_get_info, librsb), RSBErr,
                (Ptr{Void}, Cint, Ptr{Void}),
                A.ptr, RSB_MIF_MATRIX_ROWS__TO__RSB_COO_INDEX_T, out)
    check_rsb_error(err)
    return out.x
end


function rsb_get_ncols(A::SparseMatrixRSB)
    out = Ref{RSBCooIdx}()
    err = ccall((:rsb_mtx_get_info, librsb), RSBErr,
                (Ptr{Void}, Cint, Ptr{Void}),
                A.ptr, RSB_MIF_MATRIX_COLS__TO__RSB_COO_INDEX_T, out)
    check_rsb_error(err)
    return out.x
end


function Base.size(A::SparseMatrixRSB)
    return (Int(rsb_get_nrows(A)),
            Int(rsb_get_ncols(A)))
end


function Base.size(A::SparseMatrixRSB, k::Integer)
    if k == 1
        return Int(rsb_get_nrows(A))
    elseif k == 2
        return Int(rsb_get_ncols(A))
    else
        return 1
    end
end


function Base.getindex{T}(A::SparseMatrixRSB{T}, i::Integer, j::Integer)
    yptr = Ref{T}()
    iptr = Ref{RSBCooIdx}(i)
    jptr = Ref{RSBCooIdx}(j)
    err = ccall((:rsb_mtx_get_vals, librsb), RSBErr,
                (Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, RSBNnzIdx,
                 RSBFlags),
                A.ptr, yptr, iptr, jptr, 1, RSB_FLAG_FORTRAN_INDICES_INTERFACE)
    if err == -1
        return zero(T)
    end
    check_rsb_error(err)
    return yptr.x
end


function Base.setindex!{T}(A::SparseMatrixRSB{T}, value, i::Integer, j::Integer)
    yptr = Ref{T}(value)
    iptr = Ref{RSBCooIdx}(i)
    jptr = Ref{RSBCooIdx}(j)
    err = ccall((:rsb_mtx_set_vals, librsb), RSBErr,
                (Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, RSBNnzIdx,
                 RSBFlags),
                A.ptr, yptr, iptr, jptr, 1, RSB_FLAG_FORTRAN_INDICES_INTERFACE)
    check_rsb_error(err)
    return yptr.x
end


function Base.diag{T}(A::SparseMatrixRSB{T})
    v = Vector{T}(min(size(A)...))
    err = ccall((:rsb_mtx_get_vec, librsb), RSBErr,
                (Ptr{Void}, Ptr{Void}, Cint),
                A.ptr, v, RSB_EXTF_DIAG)
    check_rsb_error(err)
    return v
end


function Base.full{T}(A::SparseMatrixRSB{T})
    B = zeros(T, size(A))
    I, J, V = findnz(A)
    for (i, j, v) in zip(I, J, V)
        B[i, j] = v
    end
    return B
end


# Sparse matrix by vector multiplication
for (f, transA) in ((:A_mul_B!,  RSB_TRANSPOSITION_N),
                    (:At_mul_B!, RSB_TRANSPOSITION_T),
                    (:Ac_mul_B!, RSB_TRANSPOSITION_C))
    @eval begin
        function Base.$f{T}(y::Vector{T}, A::SparseMatrixRSB{T}, x::Vector{T})
            m, n = size(A)
            if $transA == RSB_TRANSPOSITION_N
                @assert length(x) == n
                @assert length(y) == m
            else
                @assert length(x) == m
                @assert length(y) == n
            end

            alpha = Ref{T}(one(T))
            beta  = Ref{T}(zero(T))
            err = ccall((:rsb_spmv, librsb), RSBErr,
                        (RSBTrans, Ptr{Void}, Ptr{Void}, Ptr{Void}, RSBCooIdx,
                         Ptr{Void}, Ptr{Void}, RSBCooIdx),
                        $transA, alpha, A.ptr, x, 1, beta, y, 1)
            check_rsb_error(err)
            return y
        end
    end
end


# Sparse matrix by dense matrix multiplication
for (f, transA) in ((:A_mul_B!,  RSB_TRANSPOSITION_N),
                    (:At_mul_B!, RSB_TRANSPOSITION_T),
                    (:Ac_mul_B!, RSB_TRANSPOSITION_C))
    @eval begin
        function Base.$f{T}(C::Matrix{T}, A::SparseMatrixRSB{T}, B::Matrix{T})
            ma, na = size(A)
            mb, nb = size(B)
            mc, nc = size(C)

            if $transA == RSB_TRANSPOSITION_N
                @assert na == mb
                @assert (mc, nc) == (ma, nb)
            else
                @assert ma == mb
                @assert (mc, nc) == (na, nb)
            end

            alpha = Ref{T}(one(T))
            beta  = Ref{T}(zero(T))
            err = ccall((:rsb_spmm, librsb), RSBErr,
                        (RSBTrans, Ptr{Void}, Ptr{Void}, RSBCooIdx, RSBFlags,
                         Ptr{Void}, RSBNnzIdx, Ptr{Void}, Ptr{Void}, RSBNnzIdx),
                        $transA, alpha, A.ptr, nb,
                        RSB_FLAG_WANT_COLUMN_MAJOR_ORDER, B, mb, beta, C, na)
            check_rsb_error(err)
            return C
        end
    end
end


# TODO:
# get column and row sums
# convert{S, T}(::Type{SparseMatrixRSB{S}}, ::SparseMatrixRSB{T})
# transpose / transpose!
# conj / conj!
# A_ldiv_B! / trisolve (BLAS_?ussv, BLAS_?ussm)
# + (I think we just insert elements with the right flag set if we want to add)

end



