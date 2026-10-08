"""
Combine a campaign's per-multiplier subfolders (produced by `02_lambda_sweep.jl`, one
`mult<value>/` subfolder each -- see `02_lambda_sweep_parallel.sh` for running them in
parallel) into the campaign-level aggregate: one `summary.csv`, and the two plots that only
make sense across the whole sweep, `comparison.pdf` and `lcurve.pdf`. Also writes
`fields/C_fields.jld2` so `05_diagnostics.jl` works the same as after a sequential run.

Each `mult*/` subfolder is left untouched; this only (re)writes the campaign-level files
directly in `outputs/02_lambda_sweep/<tag>/`.

Run with the campaign's tag (its directory name under outputs/02_lambda_sweep/), e.g.:
    julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/02_lambda_sweep_merge.jl \
        weertman_m15C_gsf1_ep30-5
"""

include(joinpath(@__DIR__, "common.jl"))

using CSV, DataFrames, JLD2, Printf

@assert !isempty(ARGS) "usage: 02_lambda_sweep_merge.jl <tag>  (e.g. weertman_m15C_gsf1_ep30-5)"
const TAG = ARGS[1]
const CAMPAIGN_DIR = joinpath(OUT_DIR, "02_lambda_sweep", TAG)
@assert isdir(CAMPAIGN_DIR) "no campaign directory at $(CAMPAIGN_DIR)"

const MULT_DIRS = sort([joinpath(CAMPAIGN_DIR, n) for n in readdir(CAMPAIGN_DIR)
                         if startswith(n, "mult") && isdir(joinpath(CAMPAIGN_DIR, n))])
@assert !isempty(MULT_DIRS) "no mult*/ subfolders found under $(CAMPAIGN_DIR)"
@info "Merging" n_mults=length(MULT_DIRS) CAMPAIGN_DIR

results = DataFrame()
C_fields = Dict{Float64, Matrix{Float64}}()
results_by_mult = Dict{Float64, Sleipnir.Results}()
glacier = nothing
law = nothing
λ_ref = nothing
maxC = nothing
A_VALUE = nothing
A_NAME = nothing

for dir in MULT_DIRS
    csv_path = joinpath(dir, "summary.csv")
    fields_path = joinpath(dir, "fields.jld2")
    if !isfile(csv_path) || !isfile(fields_path)
        @warn "Skipping incomplete mult subfolder (missing summary.csv or fields.jld2)" dir
        continue
    end
    row = only(eachrow(CSV.read(csv_path, DataFrame)))
    push!(results, row; promote = true)
    saved = JLD2.load(fields_path)
    C_fields[saved["mult"]] = saved["C"]
    results_by_mult[saved["mult"]] = saved["res"]
    glacier = saved["glacier"]
    law = saved["law"]
    A_VALUE = saved["A"]
    A_NAME = saved["A_name"]
    maxC = saved["maxC"]
end
sort!(results, :mult)
@assert !isempty(results) "nothing to merge"

CSV.write(joinpath(CAMPAIGN_DIR, "summary.csv"), results)
jldsave(joinpath(run_dir("02_lambda_sweep", TAG, "fields"), "C_fields.jld2");
    C_fields, results_by_mult, glacier, A = A_VALUE, A_name = A_NAME,
    multipliers = results.mult, maxC, law)
@info "Merged summary" n = nrow(results) mults = results.mult

# Same two aggregate figures 02_lambda_sweep.jl draws at the end of a sequential sweep.
# One shared colorrange/colorbar across panels, not auto-scaled per panel, so panels are
# actually comparable to each other.
mults_sorted = sort(collect(keys(C_fields)))
crange = extrema(reduce(vcat, vec.(values(C_fields))))
fig = Figure(size = (380 * length(C_fields) + 120, 460))
for (k, mult) in enumerate(mults_sorted)
    ax = Axis(fig[1, k], title = "λ_C = $(mult)·λ_ref", aspect = DataAspect())
    heatmap!(ax, C_fields[mult]; colormap = :viridis, colorrange = crange)
end
cticks = range(crange[1], crange[2]; length = 5)
Colorbar(fig[1, length(mults_sorted) + 1]; colormap = :viridis, colorrange = crange,
    label = "C", ticks = cticks, tickformat = vs -> [(@sprintf "%.3e" v) for v in vs])
Label(fig[0, :], "Inverted C vs regularization — $(RGI_ID), $(TAG)"; fontsize = 20, font = :bold)
save(joinpath(CAMPAIGN_DIR, "comparison.pdf"), fig)

fig2 = Figure(size = (900, 500))
ax = Axis(fig2[1, 1], xlabel = "reg_frac  (regularization / total loss)", ylabel = "V RMSE  (m/yr)",
    xscale = log10, title = "L-curve — $(TAG)")
scatterlines!(ax, max.(results.reg_frac, eps()), results.v_rmse; markersize = 14)
for r in eachrow(results)
    text!(ax, max(r.reg_frac, eps()), r.v_rmse; text = " ×$(r.mult)", fontsize = 11)
end
save(joinpath(CAMPAIGN_DIR, "lcurve.pdf"), fig2)

@info "Wrote campaign aggregate" CAMPAIGN_DIR
println(results)
