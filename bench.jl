
using RecursiveSparseBlocks


function run(T)
    out = open("benchmark.csv", "w")
    println(out, "method,n,p,t")

    N = 100
    for (maxn, p) in [(1000000, 1e-4)]
        for _ in 1:N
            n = rand(1:maxn)
            A = sprand(T, n, n, p)
            Arsb = SparseMatrixRSB(A)
            x = rand(T, n)

            y = A*x
            tic()
            y = A*x
            t = toc()
            println(out, "SparseMatrixCSC,", n, ",", p, ",", t)

            y = Arsb*x
            tic()
            y = Arsb*x
            t = toc()
            println(out, "SparseMatrixRSB,", n, ",", p, ",", t)
        end
    end


    close(out)
end

run(Float64)


#@time bench_csc()
#@time bench_csc()
#@profile bench_csc()
#Profile.print(C=true)

#@time bench_rsb()
#@time bench_rsb()
#@profile bench_rsb()
#Profile.print(C=true)





