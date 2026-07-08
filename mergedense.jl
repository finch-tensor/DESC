##[1, 3, 8] with a factor of 3 should become [1, 2, 3, 7, 8, 9, 22, 23, 24]
##Formula: gfm[i] ==> the range [(gfm[i] * factor) - factor + 1, gfm[i] * factor]

# gfm = [[1], [1], [1], [1], [1], [1], [1], [1]]
# P = 8
# factor = 10000

@inbounds function unwrap_dense(gfm, factor, P)
    Threads.@threads for tid in 1:P
        v = gfm[tid]
        olddim = length(v)
        resize!(v, olddim * factor)
        for i in olddim:-1:1
            val = v[i]
            base = (val - 1) * factor
            for j in factor:-1:1
                v[(i - 1) * factor + j] = base + j
            end
        end
    end
end

# unwrap_dense(gfm, factor, P)
# println(gfm)