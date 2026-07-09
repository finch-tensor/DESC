

# P = 8
# max_pos = 5
# max_idx = 5

# ptr = [[1, 5, 6, 10, 13, 14], [1, 3, 6, 8, 12, 15], [1, 4, 4, 5, 6, 8], [1, 3, 5, 9, 11, 13], [1, 3, 6, 9, 12, 14], [1, 2, 6, 7, 10, 13], [1, 2, 4, 6, 10, 13], [1, 2, 4, 5, 9, 11]]
# idx = [[1, 2, 4, 5, 5, 1, 2, 3, 4, 2, 4, 5, 5], [1, 4, 2, 3, 5, 2, 5, 1, 2, 4, 5, 1, 4, 5], [2, 3, 4, 1, 3, 2, 5], [4, 5, 3, 4, 1, 2, 3, 4, 2, 4, 1, 3], [1, 2, 1, 3, 5, 1, 3, 5, 2, 3, 4, 2, 4], [4, 1, 3, 4, 5, 3, 1, 2, 3, 1, 3, 4], [4, 2, 5, 3, 4, 1, 2, 4, 5, 2, 3, 4], [5, 1, 4, 4, 1, 2, 3, 5, 2, 3]]
# gfm = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]

# nnz = sum(length, idx)

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

@inbounds function total_pos(candidate, ptr, gfm)
    total = 0
    for proc in 1:P
        lfbr = binary_search_lb(candidate, gfm[proc], 1, length(gfm[proc]))
        lfbr < 1 && continue
        total += ptr[proc][lfbr+1] - 1
    end
    return total
end

@inbounds function total_idx(candidate, pos, ptr, idx, gfm)
    total = 0
    for proc in 1:P
        lfbr = binary_search_lb(pos, gfm[proc], 1, length(gfm[proc]))
        lfbr < 1 && continue
        total += ptr[proc][lfbr] - 1

        lo_b = ptr[proc][lfbr]
        hi_b = ptr[proc][lfbr+1] - 1
        hi_b < lo_b && continue  # empty fiber, nothing more to add

        gidx = binary_search_lb(candidate, idx[proc], lo_b, hi_b)
        if gidx < 1
            total += hi_b - lo_b + 1   #if we "own" more idx than present, claim all the indices.
        else
            total += gidx - lo_b
            if idx[proc][gidx] == candidate
                total += 1
            end
        end
    end
    return total
end

##Goal: Compute [x_l, x_h), for each processor.
##Want work(x_l) to be as close to work_lb as possible (while being above it). Vice versa with work(x_h).
@inbounds function balance_lb(max_pos, max_idx, work_lb, work_ub, ptr, idx, gfm, P)
    posxl = -1
    lo, hi = 1, max_pos
    while lo <= hi
        candidate = div(lo + hi, 2)
        total = total_pos(candidate, ptr, gfm)
        if total >= work_lb
            if total <= work_ub
                posxl = candidate
            end
            hi = candidate - 1
        else
            lo = candidate + 1
        end
    end
    
    idxxl = -1
    lo, hi = 1, max_idx
    while lo <= hi
        candidate = div(lo + hi, 2)
        total = total_idx(candidate, posxl, ptr, idx, gfm)
        if total >= work_lb
            if total < work_ub
                idxxl = candidate
            end
            hi = candidate - 1
        else
            lo = candidate + 1
        end
    end

    return posxl, idxxl
end

@inbounds function balance_ub(max_pos, max_idx, work_lb, work_ub, ptr, idx, gfm, P)
    posxh_raw = -1
    lo, hi = 1, max_pos
    while lo <= hi
        candidate = div(lo + hi, 2)
        total = total_pos(candidate, ptr, gfm)
        if total <= work_ub
            if total >= work_lb
                posxh_raw = candidate
            end
            lo = candidate + 1
        else
            hi = candidate - 1
        end
    end

    # drops trailing zero-work tail
    posxh = -1
    if posxh_raw != -1
        target = total_pos(posxh_raw, ptr, gfm)
        lo, hi = 1, max_pos
        while lo <= hi
            candidate = div(lo + hi, 2)
            total = total_pos(candidate, ptr, gfm)
            if total >= target
                posxh = candidate
                hi = candidate - 1
            else
                lo = candidate + 1
            end
        end
    end

    idxxh = -1
    lo, hi = 1, max_idx

    while lo <= hi
        candidate = div(lo + hi, 2)
        total = total_idx(candidate, posxh, ptr, idx, gfm)
        if total < work_ub
            if total >= work_lb
                idxxh = candidate
            end
            lo = candidate + 1
        else
            hi = candidate - 1
        end
    end

    return posxh, idxxh
end

@inbounds function find_split(target, max_pos, max_idx, ptr, idx, gfm, P)
    posx = -1
    lo, hi = 1, max_pos
    while lo <= hi
        candidate = div(lo + hi, 2)
        total = total_pos(candidate, ptr, gfm)
        if total >= target
            posx = candidate
            hi = candidate - 1
        else
            lo = candidate + 1
        end
    end
    posx == -1 && return (max_pos, max_idx)

    idxx = -1
    lo, hi = 1, max_idx
    while lo <= hi
        candidate = div(lo + hi, 2)
        total = total_idx(candidate, posx, ptr, idx, gfm)
        if total >= target
            idxx = candidate
            hi = candidate - 1
        else
            lo = candidate + 1
        end
    end
    return (posx, idxx)
end

# for tid in 1:P
#     println("Processor $tid:")
#     chk_nnz = fld(nnz + P - 1, P)
#     work_lb = (tid - 1) * chk_nnz
#     work_ub = min(tid * chk_nnz, nnz) ##Processor is responsible for [pos_lb, pos_ub) nonzeroes

#     if tid == 1
#         lb = (1, 1)
#     else
#         # lb = balance_lb(max_pos, 10, work_lb, work_ub, ptr, idx, gfm, P)
#         lb = find_split(work_lb, max_pos, max_idx, ptr, idx, gfm, P)
#     end

#     if tid == P
#         ub = (max_pos, max_idx)
#     else
#         # ub = balance_ub(max_pos, 10, work_lb, work_ub, ptr, idx, gfm, P)
#         split = find_split(work_ub, max_pos, max_idx, ptr, idx, gfm, P)
#         ub = split[2] > 1 ? (split[1], split[2] - 1) : (split[1] - 1, max_idx)
#     end

#     println(lb)
#     println(ub)
# end
