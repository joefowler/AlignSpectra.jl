module AlignSpectra

using HDF5, PyPlot, DynamicTimeWarp
using Dierckx, Optim

include("MonotoneSpline.jl")

export MonotoneSpline, MonotoneSplineLogLog, compose_spline,
    compose_splinelog, matchspectra

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
    bin = collect(1:length(h))
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



"""Replace a DTW path (p1,p2) with (x,y) where the x values are unique and
the y values are an average over all those p2 where p1==x."""
function uniquepath(p1::Vector, p2::Vector)
    @assert length(p1)==length(p2)

    minx = p1[1]
    x = unique(p1)
    sort!(x)
    y = zeros(Float64, length(x))
    n = zeros(Int, length(x))

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


"""Find the best cubic spline with knots at {k}, such that the Euclidean distance to all pairs
of points {x_i, y_i} is minimized and such that the spline has "natural boundary conditions"
(i.e., it has zero second derivative at the extreme knots, k[1] and k[end]).
"""
function findbestspline(x::Vector, y::Vector, k::Vector)
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


"""Find the best cubic spline according to the Dierckx.Spline1D function with
weight vector w."""
function findbestspline_numeric(x::Vector, y::Vector, w::Vector)
    Dierckx.Spline1D(x, y, w=w, bc="nearest", s=1.0)
end


"""Find the best cubic spline with knots at {k}, according to Dierckx.Spline1D with
fixed knots.
"""
function findbestspline_knots(x::Vector, y::Vector, knots::Vector)
    @show knots
    Dierckx.Spline1D(x, y, knots, bc="nearest", k=3)
end





function basicdtw(data1::Vector{Float32}, data2::Vector{Float32}) #c1::Vector{Int}, c2::Vector{Int})
    clf()
    subplot(321)
    c1,b,_=plt.hist(data1, 8000, [0,8000], histtype="step", color="r", normed=true)
    db = 0.5*(b[2]-b[1])
    binctrs = b[1:end-1]+db
    pk1 = distinctpeaks(c1, 8, 100.)
    for p in pk1
        plt.plot(binctrs[p], c1[p], "ro")
    end

    c2,_,_=plt.hist(data2, 8000, [0,8000], histtype="step", color="k", normed=true)

    d1d2_distance = DynamicTimeWarp.Distance.poissonpenalty(length(data1), length(data2))
    dist,p1,p2 = DynamicTimeWarp.fastdtw(c1, c2, 10, d1d2_distance)
    subplot(322)
    u1,u2 = uniquepath(binctrs[p1],binctrs[p2])
    plot(u1,u2,"g")
    for p in pk1
        plt.plot(binctrs[p], u2[p], "ro")
    end

    f = splinelogspace(binctrs[pk1], float(u2[pk1]), bc="extrapolate")
    #   f = MonotoneSpline(binctrs[pk1], float(u2[pk1]), bc="extrapolate")
    plot(u1, f(u1), "k")

    data1e = f(float(data1))
    subplot(323)
    c1,b,_=plt.hist(data1e, 8000, [0,8000], histtype="step", color="r", normed=true)
    c2,_,_=plt.hist(data2, 8000, [0,8000], histtype="step", color="k", normed=true)
    dist,p1,p2 = DynamicTimeWarp.fastdtw(c1, c2, 20, d1d2_distance)
    subplot(324)
    u1,u2 = uniquepath(binctrs[p1],binctrs[p2])
    plot(u1,u2,"g")
    pk1 = distinctpeaks(c1, 8, 100.)
    for p in pk1
        plt.plot(binctrs[p], u2[p], "ro")
    end

    #   @show pk1, binctrs[pk1], u2[pk1]
    g = splinelogspace(binctrs[pk1], float(u2[pk1]), bc="extrapolate")
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
function one_dtw_step(hist1::Vector, hist2::Vector, binctrs::Vector)

    pk1 = distinctpeaks(hist1, 8, 100.)

    d1d2_distance = DynamicTimeWarp.Distance.poissonpenalty(sum(hist1), sum(hist2))
    dist,p1,p2 = DynamicTimeWarp.fastdtw(hist1, hist2, 20, d1d2_distance)
    @printf("The Poisson DTW path distance: %f\n", dist)
    u1,u2 = uniquepath(binctrs[p1],binctrs[p2])
    return binctrs[pk1], float(u2[pk1])
end



function matchspectra{T<:Number}(histograms::Matrix{T}; histrange::Range=0:0.5:8000)
    N = size(histograms)[2]
    if N == 1
        dummyspline = AlignSpectra.MonotoneSplineLogLog([1,10], [1,10]; bc="extrapolate")
        return [dummyspline], histograms[:,1]
    end

    if N == 2
        binedges = collect(histrange)
        db = binedges[2]-binedges[1]
        binctrs = binedges[1:end-1]+db*0.5
        Nbins = length(binctrs)

        n1, n2 = sum(histograms[:,1]), sum(histograms[:,2])
        r,s = one_dtw_step(histograms[:,1], histograms[:,2], binctrs)
        middle = (r*n1 .+ s*n2) / (n1+n2)
        f1 = AlignSpectra.MonotoneSplineLogLog(r, middle; bc="extrapolate")
        f2 = AlignSpectra.MonotoneSplineLogLog(s, middle; bc="extrapolate")

        # Now convert histograms to equivalent values, transform them, and
        # re-histogram the result. Approximate this by random sampling for
        # allowed values in the bin (for bins with <10 samples), or by evenly
        # spreading out the values to
        combined = zeros(histograms[:,1])
        for i = 1:length(combined)
            nh = histograms[i,1]
            if nh > 0
                Nval = div(nh, 8) + zeros(Int, 8)
                for j=1:nh-sum(Nval)
                    Nval[rand(1:8)] += 1
                end
                @assert sum(Nval) == nh
                values = linspace(binedges[i]+db/16., binedges[i+1]-db/16, 8)
                fvalues = f1(values)
                for (n, fv) in zip(Nval, fvalues)
                    bnum = round(Int, div(fv-binedges[1], db))+1
                    if bnum < Nbins && bnum > 0
                        combined[bnum] += n
                    end
                end
            end

            nh = histograms[i,2]
            if nh > 0
                Nval = div(nh, 8) + zeros(Int, 8)
                for j=1:nh-sum(Nval)
                    Nval[rand(1:8)] += 1
                end
                values = linspace(binedges[i]+db/16., binedges[i+1]-db/16, 8)
                fvalues = f2(values)
                for (n, fv) in zip(Nval, fvalues)
                    bnum = round(Int, div(fv-binedges[1], db))+1
                    if bnum < Nbins && bnum > 0
                        combined[bnum] += n
                    end
                end
            end
        end
        return [f1,f2], combined
    end

    if N == 3
        curvesA, combinedA = matchspectra(histograms[:,1:2]; histrange=histrange)
        input = hcat(combinedA, histograms[:,3])
        curvesB, combined = matchspectra(input; histrange=histrange)
        f1 = compose_splinelog(curvesA[1], curvesB[1])
        f2 = compose_splinelog(curvesA[2], curvesB[1])
        f3 = curvesB[2]
        return [f1,f2,f3], combined
    end

    # Break the values vector up into equal-size groups (extra in N2 if N is odd)
    N1 = div(N,2)
    N2 = N - N1

    @assert N > 3
    histograms1 = histograms[:,1:N1]
    histograms2 = histograms[:,N1+1:end]

    curves1, combined1 = matchspectra(histograms1; histrange=histrange)
    curves2, combined2 = matchspectra(histograms2; histrange=histrange)

    @show N, typeof(combined1), eltype(combined1)
    @show N2, typeof(combined2), eltype(combined2)
    inputs = hcat(combined1, combined2)
    curves3, combined3 = matchspectra(inputs; histrange=histrange)
    finalcurves = []
    for c in curves1
        push!(finalcurves, compose_splinelog(c, curves3[1]))
    end
    for c in curves2
        push!(finalcurves, compose_splinelog(c, curves3[2]))
    end

    finalcurves, combined3
end



end # module
