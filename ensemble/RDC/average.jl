using DelimitedFiles
using Bootstrap
using Statistics

residues = [r for r in 1:132];
coups = ["HACA", "NC", "HC", "HN", "CAC", "1HACA", "2HACA"];



ave = Dict(c=>Dict() for c in coups)
err = Dict(c=>Dict() for c in coups)

for r in residues
    temp = readdlm(string(r, "_values"))
    for c in coups 
        values = [temp[n, 9] for n in 1:size(temp)[1] if string(temp[n, 3], temp[n, 6])==c]
        if length(values)>0
            bs = bootstrap(mean, values, BasicSampling(100))
            ave[c][r] = original(bs)[1]
            err[c][r] = stderror(bs)[1]
        end
    end
end

for lab in keys(ave)
    tableout = [[string(r), ave[lab][r], err[lab][r]] for r in residues if r in keys(ave[lab])]
    writedlm(string(lab, "_law.out"), tableout)
end


