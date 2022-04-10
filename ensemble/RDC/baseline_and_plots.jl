using DelimitedFiles

using StatsBase
using Optim 
using Plots

pyplot()
theme(:wong, dpi=300, framestyle=:box, leg=false, grid=false)
PyPlot.rc("text", usetex=false) # allow tex rendering
PyPlot.rc("font", family="sans-serif", weight="normal", size="18")
PyPlot.rc("axes", labelsize="18")

#read experimental data

temp = readdlm("rdc_nh_ntail.exp")

exp_data = Dict(parse(Int, split(temp[n, 1],"-")[1][1:end-1])=>temp[n, 2] for n in 1:size(temp)[1]);
exp_err = Dict(parse(Int, split(temp[n, 1],"-")[1][1:end-1])=>temp[n, 3] for n in 1:size(temp)[1]);

#read simulated data 

sim_data = readdlm("HNN_law.out");

#apply baseline

L = sim_data[end, 1]
m0 = floor((L+1)/2)
a = 0.33-0.22*(1-exp(-0.015*L))
b = 1.16*1e5/(L^4)
c = 9.8-6.14*(1-exp(-0.021*L))

by =[abs(2*b*cosh(-a*(i+1-m0))-c) for i in 1:size(sim_data)[1]];


x=sort(collect(keys(exp_data)))
y = [exp_data[el] for el in x];
dy = [exp_err[el] for el in x];

p = plot(x, y, color=:grey, seriestype=:bar, yerr=dy, linewidth=0);

x1 = sim_data[:, 1];
y1 = -sim_data[:, 2].*by;
dy1 = sim_data[:, 3];

println(y1)

x2 = x1.-2
shx = [el for el in x2 if el in x]
shy = [exp_data[el] for el in shx];
shy1 = [-sim_data[n, 2]*by[n] for n in 1:size(sim_data)[1] if Int(sim_data[n, 1]-2) in shx]

foo(p) = sum((shy-p[1]*shy1).^2)

res = Optim.optimize(foo , [1.], BFGS())
#println(res)
k = Optim.minimizer(res)[1]

plot!(p, x1.-2, k*y1, color=:orange, yerr=k*dy1, xlabel="Sequence", ylabel="1DNH (Hz)")
xlims!(p, (0, maximum(x1)))
xticks!(p, 10:10:130)

savefig(p, "rdc-sim2.pdf");


