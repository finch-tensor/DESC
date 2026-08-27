include("coalesce_fast_bounds.jl")
using Random

# Generate P sorted, internally-duplicate-free copies over a (max_pos,max_idx)
# grid, with `dup_rate` controlling how often a coordinate is shared across
# multiple copies (0 = no cross-copy duplicates, 1 = every present coordinate
# tends to appear in most copies).
function random_copies(P, max_pos, max_idx, dup_rate)
    all_coords = [(pos, idx) for pos in 1:max_pos for idx in 1:max_idx]
    copies = [Tuple{Int,Int}[] for _ in 1:P]
    for c in all_coords
        rand() < 0.6 || continue  # only some coordinates are present at all
        for p in 1:P
            if rand() < (p == 1 ? 1.0 : dup_rate)
                push!(copies[p], c)
            end
        end
    end
    for p in 1:P
        sort!(copies[p])
        @assert allunique(copies[p])
    end
    return copies
end

function owner_map(P, copies, max_pos, max_idx)
    bounds = [coalesce_fast_bounds(tid, P, copies, max_pos, max_idx) for tid in 1:P]
    owner = Dict{Tuple{Int,Int},Vector{Int}}()
    all_distinct = sort(unique(vcat(copies...)))
    for c in all_distinct
        owners = Int[]
        for tid in 1:P
            b = bounds[tid]
            if b.lo !== nothing && b.lo <= c <= b.hi
                push!(owners, tid)
            end
        end
        owner[c] = owners
    end
    return bounds, owner, all_distinct
end

function check_partition(P, copies, max_pos, max_idx; verbose=false)
    bounds, owner, all_distinct = owner_map(P, copies, max_pos, max_idx)

    # 1. every distinct coordinate owned by EXACTLY one tid
    for (c, owners) in owner
        @assert length(owners) == 1 "coordinate $c owned by $(length(owners)) consumers: $owners"
    end

    # 2. bounds are internally consistent (lo <= hi when non-empty) and
    #    ranges, sorted by tid, are strictly increasing / non-overlapping
    nonempty = [(tid, bounds[tid]) for tid in 1:P if bounds[tid].lo !== nothing]
    for (tid, b) in nonempty
        @assert b.lo <= b.hi
    end
    for i in 1:length(nonempty) - 1
        @assert nonempty[i][2].hi < nonempty[i+1][2].lo "ranges overlap/touch between tid=$(nonempty[i][1]) and tid=$(nonempty[i+1][1])"
    end

    # 3. union of ranges covers exactly all_distinct
    covered = Set{Tuple{Int,Int}}()
    for (tid, b) in nonempty
        for c in all_distinct
            if b.lo <= c <= b.hi
                push!(covered, c)
            end
        end
    end
    @assert covered == Set(all_distinct) "coverage mismatch: missing $(setdiff(Set(all_distinct), covered))"

    # 4. balance: multiplicity-count per tid vs ideal
    mult_counts = zeros(Int, P)
    for tid in 1:P
        b = bounds[tid]
        b.lo === nothing && continue
        for p in 1:P
            lo_idx = searchsortedfirst(copies[p], b.lo)
            hi_idx = searchsortedlast(copies[p], b.hi)
            mult_counts[tid] += max(0, hi_idx - lo_idx + 1)
        end
    end
    nnz_mult = sum(length, copies)
    ideal = nnz_mult / P
    max_run = maximum(vcat([0], [count(==(c), vcat(copies...)) for c in all_distinct]))
    for tid in 1:P
        # slack bounded by the largest duplicate run size (a run can't be split,
        # so worst-case one consumer absorbs up to max_run-1 extra elements
        # relative to a perfectly even split)
        @assert mult_counts[tid] <= ideal + max_run "tid=$tid got $(mult_counts[tid]), ideal=$ideal, max_run=$max_run"
    end
    @assert sum(mult_counts) == nnz_mult

    if verbose
        println("bounds=", bounds)
        println("mult_counts=", mult_counts, " ideal=", ideal, " max_run=", max_run)
    end
    return true
end

println("=== Adversarial: one coordinate duplicated across ALL P copies, sitting at the ideal cut ===")
P = 4
max_pos, max_idx = 3, 4
copies = [Tuple{Int,Int}[] for _ in 1:P]
# distinct backbone, one per copy, plus a shared hot coordinate (2,2) in every copy
for p in 1:P
    push!(copies[p], (1, p))         # unique per-copy coordinate, spreads evenly
    push!(copies[p], (2, 2))         # duplicated across every copy: hot coordinate
    push!(copies[p], (3, p))
    sort!(copies[p])
end
check_partition(P, copies, max_pos, max_idx; verbose=true)
bounds = [coalesce_fast_bounds(tid, P, copies, max_pos, max_idx) for tid in 1:P]
hot = (2, 2)
owners = [tid for tid in 1:P if bounds[tid].lo !== nothing && bounds[tid].lo <= hot <= bounds[tid].hi]
println("hot coordinate (2,2) [duplicated $(P)x] owned by exactly: ", owners)
@assert length(owners) == 1
println("PASSED adversarial hot-duplicate test\n")

println("=== Extreme: single distinct coordinate, duplicated across all P copies (everything is one hot key) ===")
P = 6
copies = [[(1,1)] for _ in 1:P]
check_partition(P, copies, 1, 1; verbose=true)
bounds = [coalesce_fast_bounds(tid, P, copies, 1, 1) for tid in 1:P]
nonempty_tids = [tid for tid in 1:P if bounds[tid].lo !== nothing]
println("only tid(s) owning the single hot coordinate: ", nonempty_tids)
@assert length(nonempty_tids) == 1
println("PASSED single-hot-key test\n")

println("=== No duplicates at all (sanity: should match old disjoint-copy behavior) ===")
P = 5
copies = [[(p, i) for i in 1:4] for p in 1:P]
check_partition(P, copies, P, 4; verbose=true)
println("PASSED disjoint sanity test\n")

println("=== Randomized stress test across duplication rates ===")
Random.seed!(7)
for trial in 1:300
    P = rand(2:8)
    max_pos = rand(2:6)
    max_idx = rand(2:6)
    dup_rate = rand([0.0, 0.2, 0.5, 0.8, 1.0])
    copies = random_copies(P, max_pos, max_idx, dup_rate)
    if sum(length, copies) == 0
        continue
    end
    check_partition(P, copies, max_pos, max_idx)
end
println("PASSED 300 randomized trials across P in 2:8, dup_rate in {0,.2,.5,.8,1}\n")

println("=== More consumers than distinct coordinates (many empty partitions expected) ===")
P = 10
copies = [[(1,1),(1,2)] , [(1,1)], [(1,2)], [Tuple{Int,Int}[] for _ in 1:7]...]
copies = Vector{Vector{Tuple{Int,Int}}}(copies)
check_partition(P, copies, 1, 2; verbose=true)
println("PASSED P > distinct-coordinate-count test\n")

println("ALL TESTS PASSED")
