include("module_sspropc.jl")
include("module_tools.jl")

using FFTW
using CSV
using DataFrames

# 时间窗口，时域点数，频域点数
dt = 1e-3;
nt = 2^12;
T = dt*nt;
t = ([i for i = 0:(nt-1)] .- (nt-1)/2)*dt;
w = wspace(T, nt);
vs = fftshift(w./(2*pi));

# 光纤长度，步径，绘图数
z = 3e-2;
nz = 6000;
nplot = 5
n1 = round(nz/nplot + 0.5)
nz = n1*nplot
dz = z/nz

# 光纤参数
lambda_0 = 1064;
beta2 = -1.31;
beta3 = -4e-2;
beta4 = -1.14e-4;
beta5 = 2.73e-7;
beta6 = -9.8e-10;
betap = [0,0,beta2,beta3,beta4,beta5,beta6];
gamma = 1.;

tr = 0.;
s = 0.;

alp = [0.,];

# 原始的连续光时序
# u0 = gaussian(t)
# u0 = sechpulse(t, 0., 0.15, 1000.)
path = "/home/bell/桌面/date.csv";
u_origin = CSV.read(path).date .+ 230;

# 频率转波长
c = 299792458;
v = c/lambda_0*1e-3;
vs_v = vs .+ v;
lambda = c./vs_v .* 1e-3;

# 噪声
lambda_c = 1064e-9;
h = 6.62607015e-34;
c = 299792458;
E_noise = h*c/lambda_c;
fai = rand(nt)*2*pi;
A_noise = sqrt(E_noise)*exp.(1im*fai);
u_noise = abs2.(ifft(A_noise));

# 连续光时序
u0 = u_origin + u_noise;

zv = (z/nplot).*[i for i = 0:(nplot)];
u = zeros(ComplexF64, length(t), length(zv));
U = zeros(Float64, length(t), length(zv));

u[:, 1] = u0;
U[:, 1] = fftshift(abs.(dt*fft(u[:, 1])/sqrt(2*pi)));

for ii = 1:nplot
    u[:, ii+1] = @time sspropc(u[:, ii], dt, dz, n1, alp, betap, gamma, tr, s);
end

for ii = range(2,stop = nplot+1)
    U[:, ii] = fftshift(abs.(dt*fft(u[:, ii])/sqrt(2*pi)));
end

u_over = abs2.(u);
U_log = zeros(Float64, length(t), length(zv));
U_over = U;
for i = 1:(nplot+1)
    U_max = maximum(U[:, i]);
    U_log[:, i]  = 20*log10.(U[:, i]/U_max);
end

U_log_out = DataFrame(U_log);
CSV.write("/home/bell/桌面/U_log.csv", U_log_out);
lambda_out = DataFrame(lambda');
CSV.write("/home/bell/桌面/lambda.csv", lambda_out);
vs_out = DataFrame(vs');
CSV.write("/home/bell/桌面/vs.csv", vs_out);
u_out = DataFrame(u_over);
CSV.write("/home/bell/桌面/u.csv", u_out);
