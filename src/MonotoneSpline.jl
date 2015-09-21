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


function monotonespline(x::Vector, y::Vector; bc="error")
    """
    Return a
    """
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

    let x=copy(x), y=copy(y), dx=copy(dx),
        m=copy(m)  # Local copy so closure isn't affected by later changes to x or y
        function thisspline(z)
            if z<x[1]
                if bc=="error"
                    throw(ArgumentError)
                elseif bc=="constant"
                    return y[1]
                else
                    return y[1]+m[1]*(z-x[1])
                end
            elseif z>x[end]
                if bc=="error"
                    throw(ArgumentError)
                elseif bc=="constant"
                    return y[end]
                else
                    return y[end]+m[end]*(z-x[end])
                end
            end

            bin = searchsortedfirst(x, z)-1
            if bin<1; bin=1; end
            t = (z-x[bin])/dx[bin]
            return (cubehermitev1(t)*y[bin] + cubehermitem1(t)*m[bin]*dx[bin] +
            cubehermitev2(t)*y[bin+1] + cubehermitem2(t)*m[bin+1]*dx[bin])
        end
    end
    @vectorize_1arg Number thisspline
    thisspline
end

function test1()
    x = [0:9]
    y = [0,1,3,10,10,11,12,14,14,16]
    f = monotonespline(x, y, bc="extrapolate")
    f2 = Dierckx.Spline1D(float(x), float(y))
    a = linspace(-2,10,500)
    clf()
    plot(x,y,"ok",label="Input data")
    plot(a, f(a), "r", label="Monotone spline")
    plot(a, evaluate(f2,a), "b", label="Dierckx.Spline1D")
    legend(loc="lower right")
end

function splinelogspace(x::Vector, y::Vector; bc="error")
    xp = x[x.>0]
    yp = y[y.>0]
    f = monotonespline(log(xp), log(yp), bc=bc)
    g(z::Number) = exp(f(log(z)))
    @vectorize_1arg Number g
    g
end


function test2()
    model(x) = 1000*((x./1000).^0.6 ) + x
    x = linspace(0,8000,12)
    y = model(x)
    f = monotonespline(x, y, bc="extrapolate")
    g = splinelogspace(x, y, bc="extrapolate")
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

# test1()
