# Adapted from Knet's src/data.jl (author: Deniz Yuret)

struct DataLoader
    data
    batchsize::Int
    nobs::Int
    partial::Bool
    imax::Int
    indices::Vector{Int}
    shuffle::Bool
end

"""
    DataLoader(data...; batchsize=1, shuffle=false, partial=true)

An object that iterates over mini-batches of `data`, each mini-batch containing `batchsize` observations
(except possibly the last one). 

Takes as input one or more data tensors, e.g. X in unsupervised learning, X and Y in 
supervised learning. The last dimension in each tensor is considered to be the observation
dimension. 

If `shuffle=true`, shuffles the observations each time iterations are re-started.
If `partial=false`, drops the last mini-batch if it is smaller than the batchsize.

The original data is preserved as a tuple in the `data` field of the DataLoader. 

Example usage:

    Xtrain = rand(10, 100)
    train_loader = DataLoader(Xtrain, batchsize=2) 
    # iterate over 50 mini-batches of size 2
    for x in train_loader
        @assert size(x) == (10, 2)
        ...
    end

    train_loader.data   # original dataset

    Xtrain = rand(10, 100)
    Ytrain = rand(100)
    train_loader = DataLoader(Xtrain, Ytrain, batchsize=2, shuffle=true) 
    for epoch in 1:100
        for (x, y) in train_loader
            @assert size(x) == (10, 2)
            @assert size(y) == (2,)
            ...
        end
    end

    # train for 10 epochs
    using IterTools: ncycle 
    Flux.train!(loss, ps, ncycle(train_loader, 10), opt)
"""
function DataLoader(data...; batchsize=1, shuffle=false, partial=true)
    length(data) > 0 || throw(ArgumentError("Need at least one data input"))
    batchsize > 0 || throw(ArgumentError("Need positive batchsize"))
    
    nx = size(data[1])[end]
    for i=2:length(data)
        nx != size(data[i])[end] && throw(DimensionMismatch("All data should contain same number of observations"))
    end
    if nx < batchsize
        @warn "Number of data points less than batchsize, decreasing the batchsize to $nx"
        batchsize = nx
    end
    imax = partial ? nx : nx - batchsize + 1
    ids = 1:min(nx, batchsize)
    DataLoader(data, batchsize, nx, partial, imax, [1:nx;], shuffle)
end

getdata(x::AbstractArray, ids...) = x[ntuple(d -> :, (ndims(x) - length(ids)))..., ids...]

viewdata(x::AbstractArray, ids...) = @view x[ntuple(d -> :, (ndims(x) - length(ids)))..., ids...]

@propagate_inbounds function Base.iterate(d::DataLoader, i=0)     # returns data in d.indices[i+1:i+batchsize]
    i >= d.imax && return nothing
    if d.shuffle && i == 0
        shuffle!(d.indices)
    end
    nexti = min(i + d.batchsize, d.nobs)
    ids = d.indices[i+1:nexti]
    if length(d.data) == 1
        batch = getdata(d.data[1], ids)
    else
        batch = ((getdata(x, ids) for x in d.data)...,)
    end
    return (batch, nexti)
end

function Base.length(d::DataLoader)
    n = d.nobs / d.batchsize
    d.partial ? ceil(Int,n) : floor(Int,n)
end

struct SequenceLoader
    data
    batchsize::Int
    seqlen::Int
    nbatch::Int
    nseq::Int
    indices::Vector{Int}
    shuffle::Bool
    format::Symbol
end

"""
    SequenceLoader(data...; batchsize=1, seqlen=0, shuffle=false, partial=true)

An object that iterates over mini-batch sequences of `data`, each mini-batch sequence containing 
`batchsize` observations of length `seqlen` (except possibly the last one). 

Takes as input one or more data tensors, e.g. X in unsupervised learning, X and Y in 
supervised learning. The last dimension in each tensor is considered to be the time
dimension, and the last but one dimension in each tensor is considered to be the observation 
dimension.

If `shuffle=true`, shuffles the observations each time iterations are re-started.
If `partial=false`, drops the last mini-batch/sequqnce if it is smaller than the batchsize/seqlen.
If `partial=false`, drops the last mini-batch/sequqnce if it is smaller than the batchsize/seqlen.

If `format=:return_sequence`, return a sequence of mini-batches of length `seqlen`.
If `format=:time_last`, return ans array of size `(..., batchsize, seqlen)`.
If `format=:batch_last`, return an array of size `(..., seqlen, batchsize)`.

The original data is preserved as a tuple in the `data` field of the SequenceLoader.
"""
function SequenceLoader(data...; batchsize = 1, seqlen = 0, shuffle = false, partial = true, format = :return_sequence)
    length(data) > 0 || throw(ArgumentError("Need at least one data input"))
    batchsize > 0 || throw(ArgumentError("Need positive batchsize"))

    no, nt = size(data[1])[(end - 1):end]
    if seqlen < 1
        seqlen = size(data[1])[end]
    end
    for i in 2:length(data)
        no != size(data[i])[end] && throw(DimensionMismatch("All data should contain same number of observations"))
        nt != size(data[i])[end] && throw(DimensionMismatch("All data should contain same number of time steps"))
    end
    if no < batchsize
        @warn "Number of observations less than batchsize, decreasing the batchsize to $no"
        batchsize = no
    end
    if nt < seqlen
        @warn "Number of time steps less than seqlen, decreasing the seqlen to $nt"
        seqlen = nt
    end
    if partial
        nseq = floor(Int, nt / seqlen)
        nbatch = floor(Int, no / batchsize)
    else
        nseq = ceil(Int, nt / seqlen)
        nbatch = ceil(Int, no / batchsize)
    end
    indices = 1:(nseq * nbatch)
    DataLoader(data, batchsize, seqlen, nbatch, nseq, indices, shuffle, format)
end

function _get_indices(d, idx)
    no, nt = size(d.data[1])[(end - 1):end]
    t, n = fldmod1(idx, d.nbatch)
    ns = (d.batchsize * n):min(no, d.batchsize * (n + 1))
    ts = (d.seqlen * t):min(nt, d.seqlen * (t + 1))
    return ns, ts
end

function _seqmap(f, xs; format = :return_sequence, copy = true)
    y = f(xs...)
    if format == :return_sequence
        y = [selectdim(y, ndims(y), n) for n in 1:size(y, ndims(y))]
        y = copy ? Base.copy.(y) : y
    elseif format == :batch_last
        y = PermutedDimsArray(y, (ndims(y), ndims(y) - 1))
        y = copy ? Base.copy(y) : y
    end
    return y
end

function Base.getindex(d::SequenceLoader, i)
    if d.shuffle && i == 0
        shuffle!(d.indices)
    end
    idx = d.indices[i]
    ids = _get_indices(d, idx)
    batch = map(d.data) do x
        _seqmap(getdata, x, ids, format = d.format)
    end
    return length(batch) == 1 ? batch[1] : batch
end

function predict!(f, y, d::SequenceLoader)
    for i in 1:length(d)
        inds = _get_indices(d, i)
        yi = ((_seqmap(viewdata, x, inds, format = d.format, copy = false) for x in y)...,)
        ŷi = f((_seqmap(getdata, x, inds, format = d.format, copy = true) for x in d.data)...)
        for (z, ẑ) in zip(yi, ŷi)
            if eltype(ẑ) <: Number
                copyto!(z, ẑ)
            else
                copyto!.(z, ẑ)
            end
        end
    end
    return y
end

predict!(f, y::AbstractArray, d::SequenceLoader) = 
    predict!((y,), (a...) -> (f(a...),), d)

Base.length(d::SequenceLoader) = length(d.indices)

function Base.split(d::SequenceLoader, r = 1.0)
    dtrn, dtst = copy(d), copy(d)
    split_at = floor(Int, length(d) * (1 - r))
    dtrn.indices = collect(1:split_at)
    dtst.indices = collect((split_at + 1):length(d))
    return dtrn, dtst
end

function seqloss(loss)
    function f(data...)
        l = n = 0f0
        for d in zip(data...)
            l += loss(data...)
            n += 1
        end
        return l / n
    end
end
