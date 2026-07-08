using DataStructures

Base.@propagate_inbounds function binary_search(target::Int, arr)
    lo = 1
    hi = length(arr)
    @assert target > 0

    if target >= arr[hi]
        return -1
    end

    while lo <= hi
        mid = div(lo + hi, 2)
        if arr[mid] <= target && arr[mid + 1] > target
            return mid
        elseif arr[mid] > target
            hi = mid
        else
            lo = mid
        end
    end

    return -1
end

Base.@propagate_inbounds function binary_search_scalar(target, arr, lo, hi)
    while lo <= hi
        mid = div(lo + hi, 2)
        if arr[mid] == target
            return mid
        elseif arr[mid] > target
            hi = mid - 1
        else
            lo = mid + 1
        end
    end
    return -1
end

P = 2
max_pos = 10
gfm = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] #implicitly sorted by task first then local fiber. 
ptr = [[1, 1, 1, 2, 3, 5, 5, 5, 6, 6, 7], [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4]]
idx = [[6, 4, 4, 5, 3, 7], [5, 3, 7]]

##We could perhaps additionally implement a load balancer
##that has each processor use binary searches amongst positions to find an even partiiton.
##For now, we assume an even partition (each thread statically assigned 1/P positions)
function merge(gfm, ptr, idx, P, max_pos)
    lvl_ptr = Vector{Int}(undef, max_pos + 1)
    lvl_ptr[1] = 1
    uq_pairs = Vector{Int}(undef, P + 1)
    uq_pairs[1] = 0
    gfm2 = Vector{Vector{Int}}(undef, P)
    for p in 1:P
        gfm2[p] = Vector{Int}(undef, length(idx[p]))
    end

    chk_size = fld(max_pos + P - 1, P)
    Threads.@threads for tid in 1:P
        uq_pairs[tid + 1] = 0
        pos = (tid - 1) * chk_size + 1 #global position to be merged
        cap = min(tid * chk_size + 1, max_pos + 1)
        heap = BinaryMinHeap{Tuple{Int, Int, Int, Int, Int}}()
        sizehint!(heap, P)

        for proc in 1:P
            lo, hi = 1, length(ptr[proc])
            lfbr = binary_search_scalar(pos, gfm[proc], lo, hi)

            adj_pos = pos
            ##Find the first present position. This may need to be replaced with a better init algorithm.
            while lfbr < 1 && adj_pos < cap
                adj_pos += 1
                lfbr = binary_search_scalar(adj_pos, gfm[proc], lo, hi)
            end
            if adj_pos >= cap
                continue
            end

            ##skip zeroes.
            while ptr[proc][lfbr + 1] - ptr[proc][lfbr] < 1 && lfbr < length(ptr[proc])
                lfbr += 1
            end
            if lfbr >= length(ptr[proc])
                continue
            end
            adj_pos = gfm[proc][lfbr]
            if adj_pos >= cap
                continue
            end

            nz = ptr[proc][lfbr]
            push!(heap, (adj_pos, idx[proc][nz], lfbr, 0, proc))
        end
        push!(heap, (typemax(Int), typemax(Int), -1, -1, -1))

        c_pos, c_idx, c_lfbr, c_nz, c_proc = pop!(heap)
        prev = (c_pos, -1)
        seen = 0

        while !isempty(heap)
            if (c_pos, c_idx) != prev
                ##New position, otherwise just a new index
                if prev[1] != c_pos
                    lvl_ptr[prev[1] + 1] = seen
                    seen = 0
                end
                seen += 1
                uq_pairs[tid + 1] += 1
                prev = (c_pos, c_idx)
            end

            delta = ptr[c_proc][c_lfbr + 1] - ptr[c_proc][c_lfbr]
            c_nz += 1

            if c_nz >= delta
                c_nz = 0
                c_lfbr += 1

                while c_lfbr < length(ptr[c_proc]) && ptr[c_proc][c_lfbr + 1] - ptr[c_proc][c_lfbr] < 1
                    c_lfbr += 1
                end
            end
            
            if c_lfbr < length(ptr[c_proc])
                c_gpos = gfm[c_proc][c_lfbr]
                c_idx = idx[c_proc][ptr[c_proc][c_lfbr] + c_nz]
                if c_gpos < cap
                    push!(heap, (c_gpos, c_idx, c_lfbr, c_nz, c_proc))
                end
            end
 
            c_pos, c_idx, c_lfbr, c_nz, c_proc = pop!(heap)
        end
        lvl_ptr[prev[1] + 1] = seen
    end

    for p in 2:P + 1
        uq_pairs[p] += uq_pairs[p - 1]
    end
    lvl_idx = Vector{Int}(undef, uq_pairs[end])


    ##Phase 2: Compute idx and gfm2.
    Threads.@threads for tid in 1:P
        pos = (tid - 1) * chk_size + 1
        cap = min(tid * chk_size + 1, max_pos + 1)
        heap = BinaryMinHeap{Tuple{Int, Int, Int, Int, Int}}()
        sizehint!(heap, P)
        ##Can probably reduce this preprocessing.
        for proc in 1:P
            lo, hi = 1, length(ptr[proc])
            lfbr = binary_search_scalar(pos, gfm[proc], lo, hi)

            adj_pos = pos
            while lfbr < 1 && adj_pos < cap
                adj_pos += 1
                lfbr = binary_search_scalar(adj_pos, gfm[proc], lo, hi)
            end
            if adj_pos >= cap
                continue
            end

            while ptr[proc][lfbr + 1] - ptr[proc][lfbr] < 1 && lfbr < length(ptr[proc])
                lfbr += 1
            end
            if lfbr >= length(ptr[proc])
                continue
            end
            adj_pos = gfm[proc][lfbr]
            if adj_pos >= cap
                continue
            end

            nz = ptr[proc][lfbr]
            push!(heap, (adj_pos, idx[proc][nz], lfbr, 0, proc))
        end
        
        push!(heap, (typemax(Int), typemax(Int), -1, -1, -1))

        c_pos, c_idx, c_lfbr, c_nz, c_proc = pop!(heap)
        prev = (c_pos, -1)
        seen = 0

        while !isempty(heap)
            if (c_pos, c_idx) != prev
                ##Every unique pair is a unique index.
                lvl_idx[uq_pairs[tid] + seen + 1] = c_idx
                seen += 1
                prev = (c_pos, c_idx)
            end
            gfm2[c_proc][ptr[c_proc][c_lfbr] + c_nz] = uq_pairs[tid] + seen

            delta = ptr[c_proc][c_lfbr + 1] - ptr[c_proc][c_lfbr]
            c_nz += 1

            if c_nz >= delta
                c_nz = 0
                c_lfbr += 1

                while c_lfbr < length(ptr[c_proc]) && ptr[c_proc][c_lfbr + 1] - ptr[c_proc][c_lfbr] < 1
                    c_lfbr += 1
                end
            end
            
            if c_lfbr < length(ptr[c_proc])
                c_gpos = gfm[c_proc][c_lfbr]
                c_idx = idx[c_proc][ptr[c_proc][c_lfbr] + c_nz]
                if c_gpos < cap
                    push!(heap, (c_gpos, c_idx, c_lfbr, c_nz, c_proc))
                end
            end
 
            c_pos, c_idx, c_lfbr, c_nz, c_proc = pop!(heap)
        end
    end
    return lvl_ptr, lvl_idx, gfm2
end

lvl_ptr, lvl_idx, gfm2 = merge(gfm, ptr, idx, P, max_pos)
println(lvl_ptr)
println(cumsum(lvl_ptr))
println(lvl_idx)
println(gfm2)