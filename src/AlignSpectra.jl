module AlignSpectra

using HDF5, PyPlot, DynamicTimeWarp
using Dierckx, Optim

include("MonotoneSpline.jl")

export monotonespline #?? Not sure what to export yet!

"""Find <npeaks> distinct peaks in a histogram with contents <h>.
The algorithm is to label the fullest bin as the 1st peak, and penalize
all other bins by P=d/(d+<distscale>), where d is the bin distance to the
nearest peak. The bin with the 2nd-highest value of P*h is the 2nd peak,
the penalties P are recomputed, and the selection step is repeated until
<npeaks> peaks are found and return as a sorted list of bin numbers.

Returns: peak_bin_numbers::Vector{eltype(h)}, a vector of length <npeaks>.
"""
function distinctpeaks(h::Vector, npeaks::Integer, distscale::Number)
    if npeaks > length(h)
        error("Histogram has only $(length(h)) bins: cannot find $(npeaks).")
    end
    bin = [1:length(h)]
    peaks = Int[indmax(h)]
    dist = abs(peaks[1] .- bin)
    penalty = dist ./ (distscale+dist)
    for pknum in 2:npeaks
        for p in peaks[2:end]
            dist = abs(p .- bin)
            penalty = min(penalty, dist./(distscale+dist))
        end
        push!(peaks, indmax(h .* penalty))
    end
    sort(peaks)
end



function uniquepath(p1::Vector, p2::Vector)
    """Replace a DTW path (p1,p2) with (x,y) where the x values are unique and
    the y values are an average over all the p2 having p1==x."""
    minx = p1[1]
    x = unique(p1)
    sort!(x)
    y = zeros(Float64, length(x))
    n = zeros(Int, length(x))
    @assert length(p1)==length(p2)

    bin = 1
    for i=1:length(p2)
        while x[bin] < p1[i]
            bin += 1
        end
        n[bin] += 1
        y[bin] += p2[i]
    end
    float(x), y./n
end


function findbestspline(x::Vector, y::Vector, k::Vector)
    """
    Find the best cubic spline with knots at {k}, such that the Euclidean distance to all pairs
    of points {x_i, y_i} is minimized and such that the spline has "natural boundary conditions"
    (i.e., it has zero second derivative at the k[1] and k[end]).
    """
    k = sort(k)
    const m = length(k)
    const n = length(x)
    @assert length(x) == length(y)

    Mdiag = ones(Float64, m)
    for i=2:m-1
        Mdiag[i] = 2.*(k[i+1]-k[i-1])
    end
    Mdu = zeros(Float64, m-1)
    Mdl = zeros(Float64, m-1)
    for i=1:m-2
        Mdu[i+1] = k[i+2]-k[i+1]
        Mdl[i] = k[i+1]-k[i]
    end
    M = Tridiagonal(Mdl, Mdiag, Mdu)

    Ndiag = zeros(Float64, m)
    for i=2:m-1
        Ndiag[i] = 6.*(1./(k[i]-k[i-1]) - 1./(k[i+1]-k[i]))
    end
    Ndl = zeros(Float64, m-1)
    Ndu = zeros(Float64, m-1)
    for i=2:m-1
        Ndl[i-1] = -6./(k[i]-k[i-1])
        Ndu[i] = +6./(k[i]-k[i-1])
    end
    N = Tridiagonal(Ndl, Ndiag, Ndu)

    # s'' = R s where s are the f(x) values at x={k} and s'' are 2nd derivatives there
    R = inv(M) * full(N)

    V = zeros(Float64, n, m)
    for i=1:n
        xi = x[i]

        # Below lowest knot
        if xi<=k[1]
            thisk = 1
            dr = k[2]-k[1]
            B=(xi-k[thisk])/dr
            A=1-B
            V[i,thisk] = A
            V[i,thisk+1] = B

            # Above highest knot
        elseif xi>=k[m]
            thisk = m-1
            dr = k[end]-k[end-1]
            B=(xi-k[thisk])/dr
            A=1-B
            V[i,thisk] = A
            V[i,thisk+1] = B

            # In the proper spline range
        else
            thisk = findn(xi.>=k)[1][1]
            if thisk>=m || thisk<1
                @show thisk, k, xi
            end
            dr = k[thisk+1]-k[thisk]
            B=(xi-k[thisk])/dr
            A=1-B
            C = (A^3-A)*(dr^2)/6.
            D = (B^3-B)*(dr^2)/6.
            V[i,thisk] = A
            V[i,thisk+1] = B
            V[i,:] += C*R[thisk,:]
            V[i,:] += D*R[thisk+1,:]
        end
    end
    knotvals = pinv(V)*y
end


function findbestspline_numeric(x::Vector, y::Vector, w::Vector)
    """
    Find the best cubic spline according to Dierckx.
    """
    Dierckx.Spline1D(x, y, w=w, bc="nearest", s=1.0)
end

function findbestspline_knots(x::Vector, y::Vector, knots::Vector)
    """
    Find the best cubic spline with knots at {k}, such that the Euclidean distance to all pairs
    of points {x_i, y_i} is minimized and such that the spline has "natural boundary conditions"
    (i.e., it has zero second derivative at the k[1] and k[end]).
    """
    @show knots
    Dierckx.Spline1D(x, y, knots, bc="nearest", k=3)
end

function basicdtw_v1(data1::Vector{Float32}, data2::Vector{Float32}) #c1::Vector{Int}, c2::Vector{Int})
    clf()
    subplot(321)
    c1,b,_=plt.hist(data1, 8000, [0,8000], histtype="step", color="r", normed=true)
    db = 0.5*(b[2]-b[1])
    pk1 = distinctpeaks(c1,8)
    for p in pk1
        plt.plot(b[p]+db, c1[p], "ro")
    end
    pk1 =  sort(vcat([2], pk1, [length(c1)-1]))

    c2,_,_=plt.hist(data2, 8000, [0,8000], histtype="step", color="k", normed=true)

    dist,p1,p2 = DynamicTimeWarp.fastdtw(c1, c2, 20)
    subplot(322)
    u1,u2 = uniquepath(p1,p2)
    plot(u1,u2,"g")
    for p in pk1
        plt.plot(b[p]+db, u2[p], "ro")
    end

    #   sval = findbestspline(u1, u2, pk1)
    #   @show pk1
    #   @show sval
    #   for i=1:length(sval)
    #     plot(pk1[i], sval[i], "or")
    #   end
    #   weights = c2/float(sum(c2))*100
    #   weights[1] = weights[end] = 100
    #   f = findbestspline_numeric(u1, u2, weights
    #   knots = float(pk1)
    knots=float([2:250:length(c1)])
    f = findbestspline_knots(u1, u2, knots)
    plot(u1, evaluate(f, u1), "k")

    data1e = evaluate(f, convert(Array{Float64,1},data1))
    subplot(323)
    c1,b,_=plt.hist(data1e, 8000, [0,8000], histtype="step", color="r", normed=true)
    c2,_,_=plt.hist(data2, 8000, [0,8000], histtype="step", color="k", normed=true)
    dist,p1,p2 = DynamicTimeWarp.fastdtw(c1, c2, 10)
    subplot(324)
    u1,u2 = uniquepath(p1,p2)
    plot(u1,u2,"g")
    knots=float([2:250:length(c1)])
    f2 = findbestspline_knots(u1, u2, knots)
    plot(u1, evaluate(f2, u1), "k")

    data1f = evaluate(f2, float(data1e))
    subplot(313)
    c1,b,_=plt.hist(data1f, 8000, [0,8000], histtype="step", color="r", normed=true)
    c2,_,_=plt.hist(data2, 8000, [0,8000], histtype="step", color="k", normed=true)
end


function showentropy(x::Vector, y::Vector)
    clf();
    subplot(211)
    c1,b,_=plt.hist(x, 8000, [0,8000], histtype="step", color="r", normed=true)
    c2,_,_=plt.hist(y, 8000, [0,8000], histtype="step", color="k", normed=true)
    subplot(212)
    e1 = -log2(c1+1e-6).*c1
    e2 = -log2(c2+1e-6).*c2
    plot(b[2:end], e1, "r", b[2:end],e2,"k")
end

function pdistfunction(M::Integer, N::Integer)
    function dist(i, j)
        if i+j == 0; return 0; end
        mu = float(i+j)/float(M+N)
        return (i-j+mu*(N-M))^2 / float(i+j)
    end
    return dist
end

function basicdtw(data1::Vector{Float32}, data2::Vector{Float32}) #c1::Vector{Int}, c2::Vector{Int})
    clf()
    subplot(321)
    c1,b,_=plt.hist(data1, 8000, [0,8000], histtype="step", color="r", normed=true)
    db = 0.5*(b[2]-b[1])
    bctr = b[1:end-1]+db
    pk1 = distinctpeaks(c1, 8, 100.)
    for p in pk1
        plt.plot(bctr[p], c1[p], "ro")
    end

    c2,_,_=plt.hist(data2, 8000, [0,8000], histtype="step", color="k", normed=true)

    d1d2_distance = DynamicTimeWarp.Distance.poissonpenalty(length(data1), length(data2))
    #   d1d2_distance =
    dist,p1,p2 = DynamicTimeWarp.fastdtw(c1, c2, 10, d1d2_distance)
    subplot(322)
    u1,u2 = uniquepath(bctr[p1],bctr[p2])
    plot(u1,u2,"g")
    for p in pk1
        plt.plot(bctr[p], u2[p], "ro")
    end

    f = splinelogspace(bctr[pk1], float(u2[pk1]), bc="extrapolate")
    #   f = monotonespline(bctr[pk1], float(u2[pk1]), bc="extrapolate")
    plot(u1, f(u1), "k")

    data1e = f(float(data1))
    subplot(323)
    c1,b,_=plt.hist(data1e, 8000, [0,8000], histtype="step", color="r", normed=true)
    c2,_,_=plt.hist(data2, 8000, [0,8000], histtype="step", color="k", normed=true)
    dist,p1,p2 = DynamicTimeWarp.fastdtw(c1, c2, 20, d1d2_distance)
    subplot(324)
    u1,u2 = uniquepath(bctr[p1],bctr[p2])
    plot(u1,u2,"g")
    pk1 = distinctpeaks(c1, 8, 100.)
    for p in pk1
        plt.plot(bctr[p], u2[p], "ro")
    end

    #   @show pk1, bctr[pk1], u2[pk1]
    g = splinelogspace(bctr[pk1], float(u2[pk1]), bc="extrapolate")
    plot(u1, g(u1), "k")

    data1f = g(float(data1e))
    subplot(313)
    c1,b,_=plt.hist(data1f, 8000, [0,8000], histtype="step", color="r", normed=true)
    c2,_,_=plt.hist(data2, 8000, [0,8000], histtype="step", color="k", normed=true)
    return data1f
end


"""Take a single DTW step to align data1 and data2.
Finds "distinct peaks" in data1 histogram (of a minimum size and separation).
Finds the FastDTW path to align the spectra (using a Poisson-inspired distance
function) via fastdtw and then "uniquifies" the path using uniquepath so that
no x value (data1 bin number) is hit twice. Finally, it returns only the
peak locations in the two spaces, but doing lookups in the unique path.

Return: (p1, p2) where p1 is a list of peak locations in the data1
histogram, and p2 is the location of these same peaks in data2, as
determined by FastDTW.

Uses a fixed histogram binning of 0:2:8000 at this point."""
function one_dtw_step(data1::Vector, data2::Vector)

    b,c1=Base.hist(data1, 0:2:8000)
    b,c2=Base.hist(data2, 0:2:8000)

    db = b[2]-b[1]
    bctr = b[1:end-1]+db*0.5
    pk1 = distinctpeaks(c1, 9, 100.)

    d1d2_distance = DynamicTimeWarp.Distance.poissonpenalty(length(data1), length(data2))
    dist,p1,p2 = DynamicTimeWarp.fastdtw(c1, c2, 20, d1d2_distance)
    @printf("The Poisson DTW path distance: %f\n", dist)
    u1,u2 = uniquepath(bctr[p1],bctr[p2])
    return bctr[pk1], float(u2[pk1])
end

function polish!(consensus::Vector, allknots, allcurves, data, i::Int=1)
    d = data[i];
    knotr, knots = allknots[i]
    f = allcurves[i]
    _,h2 = Base.hist(f(d), 0:0.5:8000)
    println(knots)
    consensus .-= h2
    mu = (float(consensus)+0.1)
    mu = mu*length(d)/sum(mu)
    logmu = log(mu)
    summu = sum(mu)
    clf(); plot(consensus, "k", label="Consensus")
    plot(h2, "b", label="Before polish")
    for knum = 1:length(knotr)
        kchoices = knots[knum] + [-2,-1.4,-1,-.75,-.5,-.25,-.1,0,.1,.25,.5,.75,1,1.4,2]
        # kchoices = [knots[knum]-1.4:.2:knots[knum]+1.4]
        cost = similar(kchoices)
        for (ik,k) = enumerate(kchoices)
            trial = copy(knots)
            trial[knum]=k
            f = splinelogspace(knotr, trial, bc="extrapolate")
            d2 = f(d)
            _,ccc=Base.hist(d2, 0:0.5:8000)
            loglike =  -summu
            for (j,lm) in zip(ccc,logmu)
                if j>0; loglike += j*(1.0+lm-log(j)); end
            end
            plot(2*k, -loglike, "ob")
            cost[ik] = -loglike
        end
        best_s = kchoices[findmin(cost)[2]]
        @show knum, best_s-knots[knum]
        knots[knum] = best_s
    end
    f = splinelogspace(knotr, knots, bc="extrapolate")
    println(knots)
    _,h3 = Base.hist(f(d), 0:0.5:8000)
    plot(h3, "r", label="After polish")
    legend(loc="best")
    # Save results as an update
    allcurves[i] = f
    allknots[i] = (knotr, knots)
    consensus .+= h3
    plot(consensus, "c")
    return consensus
end


end # module
