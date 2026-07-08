using DataStructures

@inbounds function binary_search(target::Int, arr)
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

@inbounds function binary_search_scalar(target, arr, lo, hi)
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

@inbounds function binary_search_lb(target, arr, lo, hi)
    result = -1
    while lo <= hi
        mid = div(lo + hi, 2)
        if arr[mid] >= target
            result = mid
            hi = mid - 1
        else
            lo = mid + 1
        end
    end
    return result
end

@inbounds function lowerfbr(a::Tuple{Int,Int,Int,Int}, b::Tuple{Int,Int,Int,Int})
    if a[1] < b[1]
        return true
    elseif a[1] == b[1] && a[2] < b[2]
        return true
    else
        return false
    end
end


include("./balancer.jl")

P = 2
max_pos = 10
gfm = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] #implicitly sorted by task first then local fiber. 
ptr = [[1, 1, 1, 2, 3, 5, 5, 5, 6, 6, 7], [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4]]
idx = [[6, 4, 4, 5, 3, 7], [5, 3, 7]]
lvl_ptr = Vector{Int}()
lvl_idx = Vector{Int}()

##We could perhaps additionally implement a load balancer
##that has each processor use binary searches amongst positions to find an even partiiton.
##For now, we assume an even partition (each thread statically assigned 1/P positions)
@inbounds function merge(gfm, ptr, idx, P, max_pos, max_idx, lvl_ptr, lvl_idx)
    resize!(lvl_ptr, max_pos + 1)
    lvl_ptr[1] = 1
    uq_pairs = Vector{Int}(undef, P + 1)
    uq_pairs[1] = 0
    gfm2 = Vector{Vector{Int}}(undef, P)
    nnz = 0
    for p in 1:P
        gfm2[p] = Vector{Int}(undef, length(idx[p]))
        nnz += length(idx[p])
    end
    partials = Vector{Int}(undef, P)
    prefixes = Vector{Int}(undef, P + 1)
    prefixes[1] = 1

    chk_size = fld(max_pos + P - 1, P)
    chk_nnz = fld(nnz + P - 1, P)
    Threads.@threads for tid in 1:P
        uq_pairs[tid + 1] = 0
        pos = (tid - 1) * chk_size + 1 #global position to be merged
        cap = min(tid * chk_size + 1, max_pos + 1)
        work_lb = (tid - 1) * chk_nnz
        work_ub = min(tid * chk_nnz, nnz) ##Processor is responsible for [pos_lb, pos_ub) nonzeroes
        
        if tid == 1
            lb = (1, 1)
            ub = balance_ub(max_pos, max_idx, work_lb, work_ub, ptr, idx, gfm, P)
        elseif tid == P
            lb = balance_lb(max_pos, max_idx, work_lb, work_ub, ptr, idx, gfm, P)
            ub = (max_pos, max_idx)
        else
            lb = balance_lb(max_pos, max_idx, work_lb, work_ub, ptr, idx, gfm, P)
            ub = balance_ub(max_pos, max_idx, work_lb, work_ub, ptr, idx, gfm, P)
        end
    
        pos = lb[1]
        cap = ub[1] + 1
        idxlb = lb[2]
        idxub = ub[2]

        posdata = Vector{Tuple{Int, Int, Int, Int}}(undef, P + 1)
        ord = Base.Order.Lt((i, j) -> lowerfbr(posdata[i], posdata[j]))
        heap = BinaryHeap{Int}(ord)
        sizehint!(heap, P)

        for proc in 1:P
            lo, hi = 1, length(gfm[proc])
            lfbr = binary_search_lb(pos, gfm[proc], lo, hi)

            ##skip zeroes.
            while ptr[proc][lfbr + 1] - ptr[proc][lfbr] < 1 && lfbr < length(ptr[proc]) && gfm[proc][lfbr] < cap
                lvl_ptr[gfm[proc][lfbr] + 1] = 0
                lfbr += 1
            end
            if lfbr >= length(ptr[proc])
                continue
            end
            adj_pos = gfm[proc][lfbr]
            if adj_pos >= cap
                continue
            end
            i = binary_search_lb(idxlb, idx[proc], ptr[proc][lfbr], ptr[proc][lfbr+1])
            if i < 1
                continue
            end
            posdata[proc] = (adj_pos, idx[proc][i], lfbr, i - ptr[proc][lfbr])
            push!(heap, proc)
        end
        
        posdata[end] = (typemax(Int), typemax(Int), -1, -1)
        push!(heap, P + 1)

        c_proc = pop!(heap)
        c_pos, c_idx, c_lfbr, c_nz = posdata[c_proc]
        prev = (c_pos, -1)
        seen = 0
        part = 0
        while !isempty(heap)
            if (c_pos, c_idx) != prev
                ##New position, otherwise just a new index
                if c_pos < cap - 1 || tid == P
                    if prev[1] != c_pos
                        lvl_ptr[prev[1]+1] = seen
                        seen = 0
                    end
                    seen += 1
                    uq_pairs[tid+1] += 1
                    prev = (c_pos, c_idx)
                else
                    if c_idx <= idxub
                        part += 1
                        seen += 1
                        uq_pairs[tid+1] += 1
                        prev = (c_pos, c_idx)
                    else
                        c_proc = pop!(heap)
                        c_pos, c_idx, c_lfbr, c_nz = posdata[c_proc]
                    end
                end
            end
            delta = ptr[c_proc][c_lfbr + 1] - ptr[c_proc][c_lfbr]
            c_nz += 1

            if c_nz >= delta
                c_nz = 0
                c_lfbr += 1

                while c_lfbr < length(ptr[c_proc]) && ptr[c_proc][c_lfbr + 1] - ptr[c_proc][c_lfbr] < 1 && gfm[c_proc][c_lfbr] < cap
                    lvl_ptr[gfm[c_proc][c_lfbr] + 1] = 0
                    c_lfbr += 1
                end
            end

            if c_lfbr < length(ptr[c_proc])
                c_gpos = gfm[c_proc][c_lfbr]
                c_idx = idx[c_proc][ptr[c_proc][c_lfbr] + c_nz]
                if c_gpos < cap
                    posdata[c_proc] = (c_gpos, c_idx, c_lfbr, c_nz)
                    push!(heap,  c_proc)
                end
            end
 
            c_proc = pop!(heap)
            c_pos, c_idx, c_lfbr, c_nz = posdata[c_proc]
        end

        if idxub == max_idx
            lvl_ptr[prev[1] + 1] = seen
            partials[tid] = 0
        else
            partials[tid] = part
        end
        
        for p in pos+2:cap
            lvl_ptr[p] = lvl_ptr[p] + lvl_ptr[p-1]
        end
        prefixes[tid + 1] = seen
    end

    for p in 2:P + 1
        uq_pairs[p] += uq_pairs[p - 1]
        prefixes[p] += prefixes[p - 1]
    end
    resize!(lvl_idx, uq_pairs[end])

    ##Phase 2: Compute idx and gfm2.
    Threads.@threads for tid in 1:P
        pos = (tid - 1) * chk_size + 1 #global position to be merged
        cap = min(tid * chk_size + 1, max_pos + 1)
        work_lb = (tid - 1) * chk_nnz
        work_ub = min(tid * chk_nnz, nnz) ##Processor is responsible for [pos_lb, pos_ub) nonzeroes

        if tid == 1
            lb = (1, 1)
            ub = balance_ub(max_pos, max_idx, work_lb, work_ub, ptr, idx, gfm, P)
        elseif tid == P
            lb = balance_lb(max_pos, max_idx, work_lb, work_ub, ptr, idx, gfm, P)
            ub = (max_pos, max_idx)
        else
            lb = balance_lb(max_pos, max_idx, work_lb, work_ub, ptr, idx, gfm, P)
            ub = balance_ub(max_pos, max_idx, work_lb, work_ub, ptr, idx, gfm, P)
        end

        pos = lb[1]
        cap = ub[1] + 1
        idxlb = lb[2]
        idxub = ub[2]

        posdata = Vector{Tuple{Int, Int, Int, Int}}(undef, P + 1)
        ord = Base.Order.Lt((i, j) -> lowerfbr(posdata[i], posdata[j]))
        heap = BinaryHeap{Int}(ord)
        sizehint!(heap, P)
        ##Can probably reduce this preprocessing.
        for proc in 1:P
            lo, hi = 1, length(gfm[proc])
            lfbr = binary_search_lb(pos, gfm[proc], lo, hi)

            ##skip zeroes.
            while ptr[proc][lfbr + 1] - ptr[proc][lfbr] < 1 && lfbr < length(ptr[proc]) && gfm[proc][lfbr] < cap
                lvl_ptr[gfm[proc][lfbr] + 1] = 0
                lfbr += 1
            end
            if lfbr >= length(ptr[proc])
                continue
            end
            adj_pos = gfm[proc][lfbr]
            if adj_pos >= cap
                continue
            end
            i = binary_search_lb(idxlb, idx[proc], ptr[proc][lfbr], ptr[proc][lfbr+1])
            if i < 1
                continue
            end
            posdata[proc] = (adj_pos, idx[proc][i], lfbr, i - ptr[proc][lfbr])
            push!(heap, proc)
        end
        posdata[end] = (typemax(Int), typemax(Int), -1, -1)
        push!(heap, P + 1)

        c_proc = pop!(heap)
        c_pos, c_idx, c_lfbr, c_nz = posdata[c_proc]
        prev = (c_pos, -1)
        seen = 0

        while !isempty(heap)
            if tid < P && c_pos == cap - 1 && c_idx > idxub
                c_proc = pop!(heap)
                c_pos, c_idx, c_lfbr, c_nz = posdata[c_proc]
                continue
            end
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
                    posdata[c_proc] = (c_gpos, c_idx, c_lfbr, c_nz)
                    push!(heap,  c_proc)
                end
            end
 
            c_proc = pop!(heap)
            c_pos, c_idx, c_lfbr, c_nz = posdata[c_proc]
        end
        
        if tid > 1
            lvl_ptr[pos] += partials[tid - 1]
        end
        for p in pos+1:cap
            lvl_ptr[p] += prefixes[tid]
        end
    end
    return gfm2, uq_pairs[end]
end

max_idx = max_pos
gfm2, nmax_pos = merge(gfm, ptr, idx, P, max_pos, max_idx, lvl_ptr, lvl_idx)
println(lvl_ptr)
println(lvl_idx)
println(gfm2)
println(nmax_pos)