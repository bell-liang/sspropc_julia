using  FFTW

function prodr(x, y)
    return real(x)*real(y) + imag(x)*imag(y)
end

function prodi(x, y)
    return real(x)*imag(y) - imag(x)*real(y)
end

function ssconverged(a, b, t, nt)
    num = 0.
    denom = 0.
    for jj = range(1, stop = nt)
        denom += abs2(b[jj]);
        num += (real(b[jj]) - real(a[jj])/nt)^2 + (imag(b[jj]) - imag(a[jj])/nt)^2;
    end
    return num/denom < t
end

function sspropc(u0::Array{Complex{Float64},1}, dt::Float64, dz::Float64, nz::Float64, alp::Array{Float64,1}, beta::Array{Float64,1}, gamma::Float64, tr::Float64=0, to::Float64=0, maxiter::Int64=4, tol::Float64=1e-5)

    traman = tr
    toptical = to

    nalpha = length(alp)
    nbeta = length(beta)
    nt = length(u0)

    halfstep = zeros(ComplexF64, 1, nt);
    u1 = zeros(ComplexF64, 1, nt);
    uv = zeros(ComplexF64, 1, nt);
    uhalf = zeros(ComplexF64, 1, nt);
    ufft = zeros(ComplexF64, 1, nt);

    # w = wspace(tv)
    w = zeros(1, nt);
    for ii = range(0, stop = Int(round((nt-1.1)/2)))
        w[ii+1] = 2*pi*ii/(dt*nt);
    end
    for ii = range(Int(round((nt-1.1)/2)) + 1, stop = nt-1)
        w[ii+1] = 2*pi*ii/(dt*nt) - 2*pi/dt;
    end

    # compute halfstep and initialize u0 and u1
    for jj = range(1, stop = nt)
        phase = 0
        if nbeta != nt
            fii = 1
            wii = 1
            time = 1
            for ii = range(1, stop = nbeta)
                phase += wii*beta[ii]/fii
                fii *= time
                time += 1
                wii *= w[jj]
            end
        else
            phase = beta[jj]
        end

        if nalpha == nt
            temp_alpha = alp[jj]
        else
            temp_alpha = alp[1]
        end

        halfstep[jj] = exp(-temp_alpha*dz/4)*cos(phase*dz/2)
        halfstep[jj] += -exp(-temp_alpha*dz/4)*sin(phase*dz/2)*im

        u1[jj] = u0[jj]
    end

    ################################################
    ufft = fft(u0)
    for m = range(1, stop = nz)
        uhalf = halfstep.*ufft
        uhalf = nt*ifft(uhalf)
        inner_iter = 0
        for ii = range(1, stop = maxiter)
            if traman == 0 && toptical == 0
                for jj = range(1, stop = nt)
                    phase = gamma*(real(u0[jj])^2 + imag(u0[jj])^2+ real(u1[jj])^2 + imag(u1[jj])^2)*dz/2
                    uv[jj] = (real(uhalf[jj])*cos(phase) + imag(uhalf[jj])*sin(phase))/nt
                    uv[jj] += (real(-uhalf[jj])*sin(phase) + imag(uhalf[jj])*cos(phase))/nt*im
                end
            else
                # the first point jj = 1
                jj = 1
                ua = u0[nt]
                ub = u0[jj]
                uc = u0[jj+1]
                nlp_imag = -toptical*(abs2(uc) - abs2(ua) + prodr(ub, uc) - prodr(ub, ua)) / (4*pi*dt)
                nlp_real = abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub, uc) - prodi(ub, ua))/(4*pi*dt)

                ua = u1[nt-1]
                ub = u1[jj]
                uc = u1[jj+1]
                nlp_imag += -toptical*(abs2(uc) - abs2(ua) + prodr(ub, uc) - prodr(ub, ua)) / (4*pi*dt)
                nlp_real += abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub, uc) - prodi(ub, ua))/(4*pi*dt)

                nlp_real *= gamma*dz/2
                nlp_imag *= gamma*dz/2

                uv[jj] = (real(uhalf[jj])*cos(nlp_real)*exp(nlp_imag) + imag(uhalf[jj])*sin(nlp_real)*exp(nlp_imag))/nt
                uv[jj] += (real(-uhalf[jj])*sin(nlp_real)*exp(nlp_imag) + imag(uhalf[jj])*cos(nlp_real)*exp(nlp_imag))/nt*im

                # the points jj = range(2, nt-1)
                for jj = range(2, stop = nt-1)
                    ua = u0[jj-1]
                    ub = u0[jj]
                    uc = u0[jj+1]
                    nlp_imag = -toptical*(abs2(uc) - abs2(ua) + prodr(ub, uc) - prodr(ub, ua)) / (4*pi*dt)
                    nlp_real = abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub, uc) - prodi(ub, ua))/(4*pi*dt)

                    ua = u1[jj-1]
                    ub = u1[jj]
                    uc = u1[jj+1]
                    nlp_imag += -toptical*(abs2(uc) - abs2(ua) + prodr(ub, uc) - prodr(ub, ua)) / (4*pi*dt)
                    nlp_real += abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub, uc) - prodi(ub, ua))/(4*pi*dt)

                    nlp_real *= gamma*dz/2
                    nlp_imag *= gamma*dz/2

                    uv[jj] = (real(uhalf[jj])*cos(nlp_real)*exp(nlp_imag) + imag(uhalf[jj])*sin(nlp_real)*exp(nlp_imag))/nt
                    uv[jj] += (real(-uhalf[jj])*sin(nlp_real)*exp(nlp_imag) + imag(uhalf[jj])*cos(nlp_real)*exp(nlp_imag))/nt*im
                end

                # the endpoint jj = nt - 1
                jj = nt
                ua = u0[jj-1]
                ub = u0[jj]
                uc = u0[1]
                nlp_imag = -toptical*(abs2(uc) - abs2(ua) + prodr(ub, uc) - prodr(ub, ua)) / (4*pi*dt)
                nlp_real = abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub, uc) - prodi(ub, ua))/(4*pi*dt)

                ua = u1[jj-1]
                ub = u1[jj]
                uc = u1[1]
                nlp_imag += -toptical*(abs2(uc) - abs2(ua) + prodr(ub, uc) - prodr(ub, ua)) / (4*pi*dt)
                nlp_real += abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub, uc) - prodi(ub, ua))/(4*pi*dt)

                nlp_real *= gamma*dz/2
                nlp_imag *= gamma*dz/2

                uv[jj] = (real(uhalf[jj])*cos(nlp_real)*exp(nlp_imag) + imag(uhalf[jj])*sin(nlp_real)*exp(nlp_imag))/nt
                uv[jj] += (real(-uhalf[jj])*sin(nlp_real)*exp(nlp_imag) + imag(uhalf[jj])*cos(nlp_real)*exp(nlp_imag))/nt*im
            end

            uv = fft(uv)
            ufft = uv.*halfstep
            uv = nt.*ifft(ufft)

            u1 = uv/nt

            if ssconverged(uv, u1, tol, nt)
                break
            else
                inner_iter += 1
            end
        end
        if inner_iter == maxiter
            println("转换失败.");
        end
        u0 = u1
    end

    println("done.")
    return u0
end
