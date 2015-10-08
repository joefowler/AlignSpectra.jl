using PyPlot, Dierckx

cubehermitev1(x) = 2*x^3-3*x^2+1
cubehermitem1(x) = x^3-2*x^2+x
cubehermitev2(x) = -2*x^3+3*x^2
cubehermitem2(x) = (x-1)*x^2

@vectorize_1arg Number cubehermitev1
@vectorize_1arg Number cubehermitem1
@vectorize_1arg Number cubehermitev2
@vectorize_1arg Number cubehermitem2

function _plothermite()
    clf()
    x = linspace(0,1,300)
    plot(x, cubehermitev1(x), label="v1", color="r")
    plot(x, cubehermitem1(x), label="m1", color="g")
    plot(x, cubehermitev2(x), label="v2", color="orange")
    plot(x, cubehermitem2(x), label="m2", color="b")
    legend(loc="center right")
end

"""A callable object that acts as a monotone cubic spline, with a
choice of behaviors at the boundaries."""
type MonotoneSpline
    x      ::Vector{Real}
    y      ::Vector{Real}
    m      ::Vector{Real}
    dx     ::Vector{Real}
    bc     ::String
    function MonotoneSpline(x::Vector, y::Vector, m::Vector,
        dx::Vector; bc="error")
        new(x, y, m, dx, bc)
    end
end


function MonotoneSpline(x::Vector, y::Vector; bc="error")

    n=length(x)
    @assert n==length(y)

    secantslopes = zeros(Float64, n-1)
    for i=1:n-1
        secantslopes[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
    end
    dx = x[2:end] .- x[1:end-1]
    m = zeros(Float64, n)
    m[2:n-1] = 0.5*(secantslopes[2:end] + secantslopes[1:end-1])
    m[1] = secantslopes[1]
    m[n] = secantslopes[n-1]

    for k=1:n-1
        if secantslopes[k] == 0
            m[k] = m[k+1] = 0
        else
            alpha = m[k] / secantslopes[k]
            beta = m[k+1] / secantslopes[k]
            if alpha < 0 || beta < 0
                throw(ArgumentError)
            end
            rsq = alpha^2 + beta^2
            if rsq>9
                tau = sqrt(9.0/rsq)
            else
                tau = 1.0
            end
            m[k] = tau*alpha*secantslopes[k]
            m[k+1] = tau*beta*secantslopes[k]
        end
    end
    return MonotoneSpline(x, y, m, dx; bc=bc)
end

function Base.call(spl::MonotoneSpline, z::Number)
    if z < spl.x[1]
        if spl.bc=="error"
            println("Oh no! 1 spl.bc=$(spl.bc)")
            throw(ArgumentError)
        elseif spl.bc=="constant"
            return spl.y[1]
        else
            return spl.y[1]+spl.m[1]*(z-spl.x[1])
        end
    elseif z > spl.x[end]
        if spl.bc=="error"
            println("Oh no! 2 spl.bc=$(spl.bc)")
            throw(ArgumentError)
        elseif spl.bc=="constant"
            return spl.y[end]
        else
            return spl.y[end]+spl.m[end]*(z-spl.x[end])
        end
    end

    bin = searchsortedfirst(spl.x, z)-1
    if bin<1; bin=1; end
    t = (z-spl.x[bin])/spl.dx[bin]
    return (cubehermitev1(t)*spl.y[bin] + cubehermitem1(t)*spl.m[bin]*spl.dx[bin] +
    cubehermitev2(t)*spl.y[bin+1] + cubehermitem2(t)*spl.m[bin+1]*spl.dx[bin])
end
Base.call(spl::MonotoneSpline, z::AbstractArray) = map(spl, z)


"""A callable object that acts as a monotone cubic spline in log-log space,
with a choice of behaviors at the boundaries."""
type MonotoneSplineLogLog
    x      ::Vector{Real}
    y      ::Vector{Real}
    ms     ::MonotoneSpline
    function MonotoneSplineLogLog(x::Vector, y::Vector; bc="error")
        ms = MonotoneSpline(log(x), log(y); bc=bc)
        new(x, y, ms)
    end
end

Base.call(spl::MonotoneSplineLogLog, z::Number) = exp(spl.ms(log(z)))
Base.call(spl::MonotoneSplineLogLog, z::AbstractArray) = map(spl, z)

"""Return the MonotoneSplineLogLog that has the same knots as c1 but passes
through the control points that result from applying c2 on top of c1."""
function compose_splinelog(c1::MonotoneSplineLogLog, c2::MonotoneSplineLogLog)
    x = c1.x
    y = c2(c1(x))
    MonotoneSplineLogLog(x, y; bc="extrapolate")
end

"""Return the MonotoneSpline that has the same knots as c1 but passes
through the control points that result from applying c2 on top of c1."""
function compose_spline(c1::MonotoneSpline, c2::MonotoneSpline)
    x = c1.x
    y = c2(c1(x))
    MonotoneSpline(x, y; bc="extrapolate")
end


function test1()
    x = collect(0:9)
    y = [0,1,3,10,10,11,12,14,14,16]
    f = MonotoneSpline(x, y; bc="extrapolate")
    f2 = Dierckx.Spline1D(float(x), float(y))
    a = linspace(-2,10,500)
    clf()
    plot(x,y,"ok",label="Input data")
    plot(a, f(a), "r", label="Monotone spline")
    plot(a, evaluate(f2,a), "b", label="Dierckx.Spline1D")
    legend(loc="lower right")
    return f
end

function test2()
    model(x) = 1000*((x./1000).^0.6 ) + x
    x = collect(linspace(0,8000,12))
    y = model(x)
    f = MonotoneSpline(x, y; bc="extrapolate")
    g = MonotoneSplineLogLog(x, y; bc="extrapolate")
    s(z) = evaluate(Dierckx.Spline1D(x, y), z)
    clf()
    a = linspace(0,8000,600)
    subplot(211)
    plot(a, model(a), "k", lw=2)
    plot(a, f(a), "r")
    plot(a, g(a), "b")
    plot(a, s(a), "g")
    plot(x,y,"ok")

    subplot(212)
    plot(a, model(a)-f(a), "r", label="Monotone spline")
    plot(a, model(a)-g(a), "b", label="Log-space mono spline")
    plot(a, model(a)-s(a), "g", label="Dierckx.Spline1D")
    legend()
end
