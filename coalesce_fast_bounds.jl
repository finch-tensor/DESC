#=
Prototype v2: value-partitioned load balancing for CoalesceLevel's :fast merge
path, allowing the SAME coordinate (pos, idx) to appear in multiple (any
number of) privatized copies. No two consumers may ever own the same
coordinate, so partitioning is done by COORDINATE VALUE (not raw flat
offset): consumer tid owns a contiguous, inclusive coordinate range
[lower_bound[tid], upper_bound[tid]] under lexicographic (pos, idx) order.

Balance metric: total entries-with-multiplicity (metric "b" -- i.e. the raw
read/merge workload, counting a coordinate once per copy it appears in), as
even as possible across P consumers, subject to the hard constraint that a
duplicated coordinate's entire run (across however many copies contain it)
must land entirely on one side of a boundary. When the ideal cut falls
inside such a run, the WHOLE run goes to the later (higher-tid) consumer,
and the earlier consumer's upper bound is pulled back to the last distinct
coordinate before the run.

Each tid computes its own (lower_bound, upper_bound) independently -- no
cross-processor communication -- using only:
  - P (number of privatized copies / consumers)
  - copies[1:P], each a sorted, internally-duplicate-free Vector{Tuple{Int,Int}}
    of (pos, idx) coordinates
  - max_pos, max_idx: bounds used only to encode (pos,idx) into a single
    integer key for the coarse value-domain bisection.

Complexity per tid: O(P log n) dominant, with a log(max_pos*max_idx) coarse
bisection wrapped around each O(P log n) rank evaluation (noted explicitly
below rather than swept under the rug).
=#

using Random

# ---------------------------------------------------------------------------
# Coordinate <-> integer key encoding, so we can bisect arithmetically.
# ---------------------------------------------------------------------------
@inline encode(pos, idx, max_idx) = (pos - 1) * max_idx + idx
@inline function decode(key, max_idx)
    idx = mod(key - 1, max_idx) + 1
    pos = div(key - 1, max_idx) + 1
    return (pos, idx)
end

# ---------------------------------------------------------------------------
# rank_mult(C): total # of entries across all P copies that are <= C
# (lexicographically), counted WITH multiplicity (metric b).
# ---------------------------------------------------------------------------
function rank_mult(copies, P, C)
    total = 0
    for p in 1:P
        total += searchsortedlast(copies[p], C)
    end
    return total
end

# ---------------------------------------------------------------------------
# pred_select(t): the largest REAL coordinate (present in the union of the
# P copies) whose rank_mult is <= t. Returns `nothing` if t < the rank of
# the smallest element overall (i.e. no coordinate qualifies).
#
# Two-step: (1) coarse bisection on the encoded integer key domain to find
# the boundary key (may land in a "gap" between real coordinates), then
# (2) snap down to the true largest real coordinate <= that key, by taking
# the max, across all P copies, of "largest element in this copy <= key".
# ---------------------------------------------------------------------------
function pred_select(copies, P, t, max_pos, max_idx)
    if t <= 0
        return nothing
    end

    lo, hi = 0, max_pos * max_idx  # key 0 == sentinel "below everything"
    while lo < hi
        mid = lo + fld(hi - lo + 1, 2)
        C = decode(mid, max_idx)
        if rank_mult(copies, P, C) <= t
            lo = mid
        else
            hi = mid - 1
        end
    end

    if lo == 0
        return nothing
    end

    C = decode(lo, max_idx)
    best = nothing
    for p in 1:P
        k = searchsortedlast(copies[p], C)
        if k > 0
            cand = copies[p][k]
            if best === nothing || cand > best
                best = cand
            end
        end
    end
    return best
end

# ---------------------------------------------------------------------------
# successor(X): the smallest REAL coordinate strictly greater than X across
# all P copies. X === nothing means "below everything" (find the global
# minimum). Returns `nothing` if no such coordinate exists.
# ---------------------------------------------------------------------------
function successor(copies, P, X)
    best = nothing
    for p in 1:P
        k = X === nothing ? 0 : searchsortedlast(copies[p], X)
        if k < length(copies[p])
            cand = copies[p][k + 1]
            if best === nothing || cand < best
                best = cand
            end
        end
    end
    return best
end

# ---------------------------------------------------------------------------
# Main entry point: coalesce_fast_bounds(tid, P, copies, max_pos, max_idx)
# Returns (lo=coordinate_or_nothing, hi=coordinate_or_nothing).
# `nothing` for both means this tid's partition is empty.
# ---------------------------------------------------------------------------
function coalesce_fast_bounds(tid, P, copies, max_pos, max_idx)
    @assert 1 <= tid <= P
    @assert length(copies) == P

    nnz_mult = sum(length, copies)
    base, rem = divrem(nnz_mult, P)
    cum_target(i) = i * base + min(i, rem)  # cumulative target rank after consumer i, i in 0:P

    prev_upper = pred_select(copies, P, cum_target(tid - 1), max_pos, max_idx)
    own_upper = pred_select(copies, P, cum_target(tid), max_pos, max_idx)

    if own_upper == prev_upper
        return (lo=nothing, hi=nothing)
    end

    lo = successor(copies, P, prev_upper)
    return (lo=lo, hi=own_upper)
end
