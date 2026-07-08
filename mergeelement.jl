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

# P = 2
# max_pos = 6
# gfm = [[1, 2, 3, 4, 5, 6], [4, 5, 6]]
# val = [[7.0, 10.0, 2.0, 2.0, 4.0, 19.0], [1.0, 3.0, 12.0]]
# lvl_val = Vector{Float64}()

@inbounds function merge_element(gfm, val, max_pos, P, lvl_val)
    resize!(lvl_val, max_pos)
    chk = fld(max_pos + P - 1, P)

    Threads.@threads for tid in 1:P
        pos_start = (tid - 1) * chk + 1
        pos_stop = min(tid * chk, max_pos)
        if pos_start > max_pos
            continue
        end
        
        for p in pos_start:pos_stop
            lvl_val[p] = 0
        end

        for proc in 1:P
            lo, hi = 1, length(gfm[proc])
            lfbr = binary_search_lb(pos_start, gfm[proc], lo, hi)

            ##Can prove the processor doesn't contain the range.
            if lfbr < 1
                continue
            end

            curr = gfm[proc][lfbr]

            while lfbr <= length(val[proc]) && (curr = gfm[proc][lfbr]) <= pos_stop
                @fastmath lvl_val[curr] += val[proc][lfbr]
                lfbr += 1
            end
        end

    end
end

# merge_element(gfm, val, max_pos, P, lvl_val)

# println(lvl_val)