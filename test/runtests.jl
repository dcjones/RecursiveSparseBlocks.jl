
using Base.Test
using RecursiveSparseBlocks

@testset "SparseMatrixCSC round trip" begin
    for T in [Float32, Float64, Complex64, Complex128]
        A = sprand(T, 1000, 1000, 0.1)
        B = SparseMatrixRSB(A)
        @test convert(SparseMatrixCSC, B) == A
    end
end


@testset "Basic properties" begin
    for T in [Float32, Float64, Complex64, Complex128]
        A = sprand(T, 1000, 1000, 0.1)
        B = SparseMatrixRSB(A)
        @test size(A) == size(B)
        @test nnz(A) == nnz(B)
        @test diag(A) == diag(B)

        AI, AJ, AV = findnz(A)
        BI, BJ, BV = findnz(A)

        ap = sortperm(AI)
        AI = AI[ap]
        AJ = AJ[ap]
        AV = AV[ap]

        bp = sortperm(BI)
        BI = BI[bp]
        BJ = BJ[bp]
        BV = BV[bp]

        @test AI == BI
        @test AJ == BJ
        @test AV == BV
    end
end


@testset "Full conversion" begin
    for T in [Float32, Float64, Complex64, Complex128]
        A = sprand(T, 100, 100, 0.1)
        B = SparseMatrixRSB(A)
        @test full(A) == full(B)
    end
end


@testset "Indexing" begin
    # getindex
    n = 500
    cnt = 100
    indexes = [(rand(1:n), rand(1:n)) for _ in 1:cnt]
    for T in [Float32, Float64, Complex64, Complex128]
        A = sprand(T, n, n, 0.5)
        B = SparseMatrixRSB(A)
        for index in indexes
            @test A[index...] == B[index...]
        end
    end

    # setindex
    for T in [Float32, Float64, Complex64, Complex128]
        A = sprand(T, n, n, 0.5)
        B = SparseMatrixRSB(A)
        I, J, V = findnz(A)

        for (i, j) in zip(I, J)
            r = rand()
            A[i, j] = r
            B[i, j] = r
        end

        @test A == convert(SparseMatrixCSC, B)
    end
end


@testset "Multiplication" begin
    # sparse matrix by dense vector
    for T in [Float32, Float64, Complex64, Complex128]
        n = 1000
        A = sprand(T, n, n, 0.1)
        B = SparseMatrixRSB(A)
        x = rand(T, n)
        ay = A*x
        by = B*x

        @test_approx_eq_eps maximum(abs(ay .- by)) zero(T) 1e-4
    end

    for T in [Float32, Float64, Complex64, Complex128]
        n = 1000
        k = 100
        A = sprand(T, n, n, 0.1)
        B = SparseMatrixRSB(A)
        X = rand(T, (n, k))

        AX = A*X
        BX = B*X

        @test_approx_eq_eps maximum(abs(AX .- BX)) zero(T) 1e-4
    end
end

