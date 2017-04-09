
using Base.Test
using RecursiveSparseBlocks

@testset "SparseMatrixCSC round trip" begin
    for T in [Float32, Float64, Complex64, Complex128]
        A = sprand(T, 1000, 1000, 0.1)
        B = SparseMatrixRSB(A)
        @test size(A) == size(B)
        @test nnz(A) == nnz(B)
        @test convert(SparseMatrixCSC, B) == A
    end
end


