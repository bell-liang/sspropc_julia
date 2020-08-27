function gaussian(t,t0=0,FWHM=1,P0=1,m=1,C=0)
    return sqrt(P0).*exp.(-((1+1im*C)/2)*(2 .*(t.-t0)./FWHM).^(2*m))
end

function sechpulse(t,t0::Float64=0.,FWHM::Float64=1.,P0::Float64=1.,C::Int64=0)
    T0 = FWHM/(2*acosh(sqrt(2)))
    return sqrt(P0)*sech.((t.-t0)/T0).*exp.(-1im*C*(t.-t0).^2/(2*T0^2))
end

function wspace(t,nt=0)
    if nt == 0
        nt = length(t)
        dt = t[2] - t[1]
        t = t[nt] - t[1] + dt
    else
        dt = t/nt
    end
    w = zeros(1, nt)
    for ii = range(0, stop = Int(round((nt-1.1)/2)))
        w[ii+1] = 2*pi*ii/(dt*nt)
    end
    for ii = range(Int(round((nt-1.1)/2)) + 1, stop = nt-1)
        w[ii+1] = 2*pi*ii/(dt*nt) - 2*pi/dt
    end
    return w
end
