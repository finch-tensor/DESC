using Finch

P = 2
max_idx = 10
max_pos = 10
# ptr = [[1, 1, 1, 1, 2, 5, 5, 5, 5, 5, 5], [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]]
ptr = [[1, 2, 5], [1, 2]]
idx = [[4, 3, 4, 5], [6]]
# pos_offsets = [[4, 8], [4, 8]]
pos_offsets = [[1, 3], [1, 3]]
lvl_ptr = Vector{Int}(undef, 4)
lvl_idx = Vector{Int}(undef, 5)
lvl_ptr[1] = 1
lvl_ptr[end] = 6
val = [[3, 8, 1, 1], [9]]
lvl_val = Vector{Float64}(undef, 5)

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

function setup_coalesce!(lvl::SparseListLevel, max_pos, coalescent)
    lvl_ptr = coalescent.ptr
    lvl_idx = coalescent.idx
    nnz = sum(length, lvl.idx.data)
    resize!(lvl_idx, nnz)
    resize!(lvl_ptr, max_pos) ##maybe need fill 0

    lvl_ptr[1] = 1
    lvl_ptr[end] = nnz + 1
    
    setup_coalesce!(lvl.lvl, nnz, coalescent.lvl)
end

function setup_coalesce!(lvl::DenseLevel, max_pos, coalescent)
    setup_coalesce!(lvl.lvl, max_pos * lvl.shape, coalescent.lvl)
end

function setup_coalesce!(lvl::ElementLevel, max_pos, coalescent)
    resize!(lvl.val.data, max_pos)
end

function setup_coalesce!(lvl::CoalesceLevel, max_pos, coalescent)
    setup_coalesce!(lvl.lvl, max_pos, coalescent)
end

function merge_splist(tid, ptr, idx, P, lvl_ptr, lvl_idx, pos_offsets, was_dense)
    nnz_cutoffs = Vector{Int}(undef, P + 1)
    nnz_cutoffs[1] = 1
    for p in 2:P+1
        nnz_cutoffs[p] = nnz_cutoffs[p - 1] + length(idx[p - 1])
    end
    nnz = nnz_cutoffs[end] - 1
    max_pos = length(lvl_ptr) - 1

    base, rem = divrem(nnz, P)
    offset = (tid - 1) * base + min(tid - 1, rem)
    chunksize = base + (tid <= rem ? 1 : 0)
    work_lb = 1 + offset

    proc_id_lower = binary_search(work_lb, nnz_cutoffs)
    nz_id_lower = work_lb - nnz_cutoffs[proc_id_lower] + 1

    if was_dense
        ##Optimize this pass, currently O(P * npos / P), way too slow.
        proc = proc_id_lower
        idx_read = nz_id_lower
        idx_write = nnz_cutoffs[proc] + nz_id_lower - 1
        ceil = idx_write + chunksize
        while idx_write < ceil
            lvl_idx[idx_write] = idx[proc][idx_read]
            idx_read += 1
            idx_write += 1

            if idx_read > length(idx[proc])
                idx_read = 1
                proc += 1
            end
        end

        pos_base, pos_rem = divrem(max_pos - 1, P)
        pos_offset = (tid - 1) * pos_base + min(tid - 1, pos_rem)
        pos_chunksize = pos_base + (tid <= pos_rem ? 1 : 0)
        pos_lb = 2 + pos_offset
        pos_ub = pos_lb + pos_chunksize - 1

        for pos in pos_lb:pos_ub
            total = 1
            for p in 1:P
                total += ptr[p][pos] - 1
            end
            lvl_ptr[pos] = total
        end
    else
        work_ub = work_lb + chunksize - 1
        proc_id_upper = binary_search(work_ub, nnz_cutoffs)
        nz_id_upper = work_ub - nnz_cutoffs[proc_id_upper] + 1
        lfbr_lower = binary_search(nz_id_lower, ptr[proc_id_lower])
        lfbr_upper = binary_search(nz_id_upper, ptr[proc_id_upper])

        pos_lb = pos_offsets[tid][proc_id_lower] + lfbr_lower - 1
        pos_ub = min(pos_offsets[tid][proc_id_upper] + lfbr_upper - 1, max_pos - 1)

        if nz_id_upper < ptr[proc_id_upper][lfbr_upper + 1] - 1
            shares_border = true
        elseif lfbr_upper < length(ptr[proc_id_upper]) - 1
            shares_border = false
        elseif proc_id_upper < P
            shares_border = pos_offsets[tid][proc_id_upper + 1] == pos_ub
        else
            shares_border = false
        end

        ##copy idx
        proc = proc_id_lower
        idx_read = nz_id_lower
        idx_write = nnz_cutoffs[proc] + nz_id_lower - 1
        ceil = idx_write + chunksize
        while idx_write < ceil
            lvl_idx[idx_write] = idx[proc][idx_read]
            idx_read += 1
            idx_write += 1

            if idx_read > length(idx[proc])
                idx_read = 1
                proc += 1
            end
        end

        ##copy pos
        proc = proc_id_lower
        pos_read = lfbr_lower

        pos_write = 2
        for p in 1:proc - 1
            pos_write += length(ptr[p]) - 1
            pos_offsets[tid][p + 1] == pos_offsets[tid][p] + length(ptr[p]) - 2 && (pos_write -= 1)
        end
        pos_write += lfbr_lower - 1

        ceil = 3
        for p in 1:proc_id_upper - 1
            ceil += length(ptr[p]) - 1
            pos_offsets[tid][p + 1] == pos_offsets[tid][p] + length(ptr[p]) - 2 && (ceil -= 1)
        end
        ceil += lfbr_upper - 1
        shares_border && (ceil -= 1)

        prefix = ptr[proc][pos_read] + nnz_cutoffs[proc] - 1
        while pos_write < ceil
            delta = ptr[proc][pos_read + 1] - ptr[proc][pos_read ]
            prefix += delta
            lvl_ptr[pos_write] = prefix
            pos_write += 1
            pos_read += 1

            if pos_read > length(ptr[proc]) - 1
                pos_read = 1
                old_proc = proc
                proc += 1
                if proc > P
                    break
                end
                if pos_offsets[tid][old_proc + 1] == pos_offsets[tid][old_proc] + length(ptr[old_proc]) - 2
                    pos_write -= 1
                end
            end
        end
        for p in 1:P
            pos_offsets[tid][p] = nnz_cutoffs[p]
        end
    end
end

function merge_element(tid, val, P, lvl_val, was_dense)
    nnz_cutoffs = Vector{Int}(undef, P + 1)
    nnz_cutoffs[1] = 1
    for p in 2:P+1
        nnz_cutoffs[p] = nnz_cutoffs[p - 1] + length(val[p - 1])
    end
    nnz = nnz_cutoffs[end] - 1

    base, rem = divrem(nnz, P)
    offset = (tid - 1) * base + min(tid - 1, rem)
    chunksize = base + (tid <= rem ? 1 : 0)
    work_lb = 1 + offset
    work_ub = work_lb + chunksize - 1

    proc_id_lower = binary_search(work_lb, nnz_cutoffs)
    nz_offset = work_lb - nnz_cutoffs[proc_id_lower] + 1
    proc = proc_id_lower
    write_idx = work_lb
    while write_idx <= work_ub
        lvl_val[write_idx] = val[proc][nz_offset]
        write_idx += 1
        nz_offset += 1
        if nz_offset > length(val[proc])
            proc += 1
            nz_offset = 1
        end
    end
end

# merge_splist(1, ptr, idx, P, lvl_ptr, lvl_idx, pos_offsets, false)
# merge_splist(2, ptr, idx, P, lvl_ptr, lvl_idx, pos_offsets, false)
# merge_element(1, val, P, lvl_val, false)
# merge_element(2, val, P, lvl_val, false)

Threads.@threads for tid in 1:P
    merge_splist(tid, ptr, idx, P, lvl_ptr, lvl_idx, pos_offsets, false)
    merge_element(tid, val, P, lvl_val, false)
end

println(lvl_ptr)
println(lvl_idx)
println(lvl_val)
println(pos_offsets)