# P = 8
# max_idx = 10
# max_pos = 10
# gfm = [[2, 3, 4, 5, 7, 8], [1, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 7, 9], [1, 2, 4, 6, 9, 10], 
#     [1, 3, 6, 7, 9], [2, 4, 5, 6, 7, 8, 9, 10], [4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 7, 9, 10]]
# ptr = [[1, 2, 4, 6, 7, 10, 14], [1, 4, 5, 6, 7, 9, 10, 12, 13, 16], [1, 2, 4, 5, 6, 7, 9, 13], [1, 2, 4, 6, 8, 12, 14],
#      [1, 2, 5, 7, 9, 12], [1, 2, 4, 7, 9, 10, 11, 12, 13], [1, 2, 5, 7, 9, 10, 11, 12], [1, 5, 7, 8, 10, 11, 15, 16, 18]]
# idx = [[1, 6, 7, 4, 6, 2, 2, 5, 10, 5, 6, 7, 10], [1, 7, 9, 1, 2, 8, 3, 5, 1, 1, 5, 4, 3, 4, 9], [5, 6, 9, 3, 8, 6, 3, 4, 2, 6, 8, 10], [1, 8, 10, 4, 9, 3, 6, 2, 4, 7, 10, 3, 10], 
#     [3, 1, 2, 10, 5, 8, 1, 5, 2, 5, 9], [7, 3, 10, 1, 2, 10, 3, 8, 4, 5, 10, 10], [9, 3, 4, 6, 5, 6, 3, 9, 4, 9, 10],
#     [3, 5, 6, 8, 1, 8, 6, 2, 7, 10, 1, 2, 6, 10, 5, 4, 7]]

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
