using Finch
using BenchmarkTools
using SparseArrays

const P = 8
const n = 10000
const dims = (n, n)


function setup_data(dims, P)
    tensors = Vector{Tensor}(undef, P)
    data = Vector{Tuple}(undef, P)
    for i in 1:P
        t = fsprand(dims[1], dims[2], 0.001)
        tensors[i] = Tensor(Dense(SparseDict(Element(0.0))), t)
        data[i] = ffindnz(t)
    end

    current_dim3 = 2
    I = data[1][1:length(dims)]
    I = (I..., fill(1, length(I[1])))
    V = data[1][length(data[1])]

    for i in 2:P
        cur_tens = (data[i][1:length(data[i]) - 1]..., fill(current_dim3, length(data[i][1])))
        I = map(vcat, I, cur_tens)
        V = vcat(V, data[i][length(data[i])])
        current_dim3 += 1
    end

    data_out = fsparse(I..., V)
    return tensors, data_out
end

println("Initializing Data... (Dense, Sparse) test")
tensors, merged_tensors = setup_data(dims, P)

@info "Warmup and Init"
A = Tensor(Dense(Dense(SparseDict(Element(0.0)))), merged_tensors)
B = Tensor(Dense(SparseDict(Element(0.0))), dims[1], dims[2])

eval(@finch_kernel function serial_merge_benchmark(A, B)
    B .= 0
    for p in _, j in _, i in _
        B[i, j] += A[i, j, p]
    end
end)

serial_merge_benchmark(A, B)

tot = nnz(SparseMatrixCSC(B))
println("total nnz: $tot")


A = Tensor(Dense(Dense(SparseList(Element(0.0)))), merged_tensors)
B = Tensor(Dense(SparseList(Element(0.0))), B)

ptr = Vector{Vector{Int64}}(undef, P)
idx = Vector{Vector{Int64}}(undef, P)
val = Vector{Vector{Float64}}(undef, P)

for i in 1:P
    current = tensors[i]

    ptr[i] = current.lvl.lvl.ptr

    idx[i] = current.lvl.lvl.idx

    val[i] = current.lvl.lvl.lvl.val
end

println("Benchmarking Parallel Algorithm:")

include("./mergesplist.jl")
include("./mergeelement.jl")
include("./mergedense.jl")

function merge_parallel(gfm, ptr, idx, P, max_pos, max_idx, lvl_ptr, lvl_idx, val, lvl_val)
    unwrap_dense(gfm, max_idx, P)
    gfm2, max_pos2 = merge(gfm, ptr, idx, P, max_pos, lvl_ptr, lvl_idx)
    merge_element(gfm2, val, max_pos2, P, lvl_val)
end

max_dim = n

resp = @benchmark merge_parallel(gfm, ptr, idx, $P, $max_dim, $n, lvl_ptr, lvl_idx, $val, lvl_val) setup=(
    gfm = [[1] for _ in 1:$P];
    lvl_ptr = Vector{Int}();
    lvl_idx = Vector{Int}();
    lvl_val = Vector{Float64}()
) evals=1

display(resp)

lvl_ptr = Vector{Int}()
lvl_idx = Vector{Int}()
lvl_val = Vector{Float64}()
global_fbr_map0 = [[1] for _ in 1:P]

merge_parallel(global_fbr_map0, ptr, idx, P, max_dim, n, lvl_ptr, lvl_idx, val, lvl_val)

@assert lvl_ptr == B.lvl.lvl.ptr
@assert lvl_idx == B.lvl.lvl.idx
@assert lvl_val == B.lvl.lvl.lvl.val

println("Benchmarking baseline:")
include("./baseline.jl")

function merge_baseline(global_fbr_map, local_fbr_map, task_map, ptr, index, val, P, lvl_ptr, lvl_idx, lvl_val, max_dim, max_idx)
    global_fbr_map, local_fbr_map, task_map = unroll_dense_coalesce(global_fbr_map, local_fbr_map, task_map, max_idx, P)
    cutoffs = compute_proc_cutoffs(index, P)
    merged_positions, merged_indices, local_fbr_map2, task_map2 = gen_pos_idx_map(global_fbr_map, local_fbr_map, task_map, ptr, index, cutoffs, P)
    global_fbr_map, local_fbr_map, task_map = process_next_lvl(merged_positions, merged_indices, task_map2, local_fbr_map2, P, max_dim, lvl_ptr, lvl_idx)
    merge_element_level(global_fbr_map, local_fbr_map, task_map, val, P, lvl_val)
end

resb = @benchmark merge_baseline(global_fbr_map, local_fbr_map, task_map, $ptr, $idx, $val, $P, lvl_ptr, lvl_idx, lvl_val, $max_dim, $n) setup=(
    global_fbr_map = fill(1, $P);
    local_fbr_map  = fill(1, $P);
    task_map       = repeat(1:$P, 1);
    lvl_ptr        = Vector{Int}();
    lvl_idx        = Vector{Int}();
    lvl_val        = Vector{Float64}()
) evals=1

display(resb)

lvl_ptr = Vector{Int}()
lvl_idx = Vector{Int}()
lvl_val = Vector{Float64}()

global_fbr_map0 = fill(1, P)
local_fbr_map0 = fill(1, P)
task_map0 = repeat(1:P, 1)
max_dim = n

merge_baseline(global_fbr_map0, local_fbr_map0, task_map0, ptr, idx, val, P, lvl_ptr, lvl_idx, lvl_val, max_dim, n)

@assert lvl_ptr == B.lvl.lvl.ptr
@assert lvl_idx == B.lvl.lvl.idx
@assert lvl_val == B.lvl.lvl.lvl.val

@info "all equivalency checks passed!"