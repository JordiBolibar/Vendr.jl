"""
Phase 2 — regularization sweep for the gridded C inversion on Aletsch.

A is pinned to one A_LADDER rung (temperate by default) and only λ_C varies, so the
smoothness of the recovered C can be picked visually before running the full A ladder.
λ_C is expressed as a multiple of a reference weight computed from the C scale of the
sliding law *and* the A rung in use, which keeps the sweep comparable across both.

At temperate A, deformation alone can already match or exceed observed velocities on
Aletsch's trunk (2A(ρg)ⁿH⁴∇S² easily exceeds 100 m/yr at H=300m for realistic slopes), which
leaves the data term no reason to push C away from zero: the first two runs of this sweep
(temperate, all six multipliers) left C_mean within ~1.3x of its seed throughout, regardless
of λ_C -- the sweep wasn't actually probing a C-vs-fit tradeoff. Pin a colder rung (see the
earlier `03_A_ladder.jl` run for which one gave C room to do real work) before reading
anything into the resulting L-curve.

Each λ writes its PDF and appends to the CSV as soon as it finishes, so partial results are
usable while the sweep runs.

Run with, law and A rung defaulting to weertman/temperate:
    julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/02_lambda_sweep.jl [weertman|budd] [gsf] [epochs] [mults] [A_name]
"""

include(joinpath(@__DIR__, "common.jl"))

using CSV
using DataFrames
using JLD2

const LAW_NAME = isempty(ARGS) ? :weertman : Symbol(ARGS[1])
const LAW = getproperty(SLIDING_LAWS, LAW_NAME)

# Resolution and epoch budget are arguments so the sweep can be run cheaply first and at
# production settings later without editing the script. Phase 2 answers a question about the
# *relative* weight of the regularization, so a coarse grid is a legitimate way to locate the
# knee of the L-curve; Phase 3 is where the resolution has to be right. Full resolution here
# costs roughly 24x the forward solve of gsf 3 (outputs/00b_perf_profile/perf_profile.csv), which puts the
# six multipliers at ~100 h.
const GSF = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
const EPOCHS = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ",")) : [10, 20]
const σ_H = 18.0    # m, from the glathida repeat surveys
# σ_V is set from glacier.velocityData.vabs_error below, once the glacier is loaded --
# real Millan22 uncertainty if the cache has it, a labelled placeholder otherwise.
const H_REF = 300.0 # m, representative trunk thickness used to set the C and H₀ scales

# Spans four orders of magnitude: λ_ref fixes the scaling between laws correctly, but its
# absolute value rests on the unknown normalization inside L2Sum, so the sweep is wide.
const MULTIPLIERS = length(ARGS) >= 4 ? parse.(Float64, split(ARGS[4], ",")) :
                   [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]

# A rung to pin for this sweep, from the shared A_LADDER (common.jl). Defaulting to
# temperate keeps old invocations reproducible, but see the module docstring above: at
# temperate A, deformation alone tends to already saturate the fit, so λ_C has nothing to
# trade off against. Pass a colder rung ("m5C".."m20C") to sweep in a regime where C matters.
const A_NAME = length(ARGS) >= 5 ? ARGS[5] : "temperate"
const A_VALUE = A_from_name(A_NAME)

# Tagged with all four so an exploratory run cannot overwrite a production one; the tag is
# the campaign's directory name (see `run_dir` in common.jl). One campaign directory holds
# one subfolder per multiplier (self-contained: its own summary.csv/fields.jld2/plot.pdf, so
# several single-multiplier processes can write concurrently -- see
# `02_lambda_sweep_parallel.sh` -- without racing each other), plus the aggregate
# summary.csv/comparison.pdf/lcurve.pdf across whichever multipliers have completed.
# The aggregate is written directly by this process only when it itself ran more than one
# multiplier (the sequential case, one writer); after a parallel run, `02_lambda_sweep_merge.jl`
# produces it by combining the per-multiplier subfolders.
const TAG = "$(LAW_NAME)_$(A_NAME)_gsf$(GSF)_ep$(join(EPOCHS, "-"))"
const CAMPAIGN_DIR = run_dir("02_lambda_sweep", TAG)
@assert length(EPOCHS) == 2 "epochs must match the two optimizers in build_params, e.g. 10,20"

# C at which sliding equals deformation, which sets both the bound and the regularization
C_scale = Huginn.sliding_scale(A_VALUE, H_REF; p = LAW.p, q = LAW.q)
maxC = 8 * C_scale   # headroom to ~89% sliding before the tanh bound bites

@info "Sliding law" LAW_NAME p=LAW.p q=LAW.q A_NAME A_VALUE C_scale maxC

probe = build_params(; tspan = (2009.0, 2018.0), t₀ = 2009.0, maxC = maxC,
    gridScalingFactor = GSF)
glacier, series = prepare_glacier(probe)

tspan = aligned_tspan(first(series.t), 2018.0, 1.0/12.0)
t₀ = first(tspan)
@info "Window" tspan campaigns=series.t
@assert all(tspan[1] .<= series.t .<= tspan[2]) "A glathida campaign falls outside the simulation window"

glacier = Sleipnir.Glacier2D(glacier; thicknessData = series)
glaciers = set_sliding_law!([glacier], LAW, 0.1 * maxC)

ncells = prod(size(glacier.H₀) .- 1)
λ_ref = reference_λ(C_scale, glacier.Δx, ncells)
# Lisa's tuned value, kept fixed so the sweep varies λ_C alone. The computed reference is
# logged for comparison and revisited once λ_C is settled.
λ_H₀ = 1e-4

@info "Weights" λ_ref λ_H₀ λ_H₀_computed=reference_λ(H_REF, glacier.Δx, prod(size(glacier.H₀)); target = 0.05) tspan grid=size(glacier.H₀)

results = DataFrame(
    law = String[], mult = Float64[], λ_C = Float64[], loss = Float64[],
    C_mean = Float64[], C_median = Float64[], C_max = Float64[], frac_at_bound = Float64[],
    roughness = Float64[], reg_C = Float64[], reg_frac = Float64[],
    slide_frac_median = Float64[],
    v_rmse = Float64[], h_rmse = Float64[], seconds = Float64[],
)

C_fields = Dict{Float64, Matrix{Float64}}()
# One Results per multiplier, so `05_diagnostics.jl` can render the observation-comparison
# and histogram plots later without re-running the sweep.
results_by_mult = Dict{Float64, Sleipnir.Results}()

V_ref = only(glacier.velocityData.vabs)
mask_V = V_ref .> 0.0

# Real Millan22 uncertainty when the glacier's cache has it (mean over ice-covered cells of
# the per-pixel err_vx/err_vy field -- see Sleipnir's _process_Millan22_error). Falls back to
# a labelled placeholder, rather than erroring, for any glacier whose gridded_data.nc predates
# requesting the error bands (OGGM's add_error flag) or hasn't been backfilled yet.
verr = glacier.velocityData.vabs_error
σ_V = if isnothing(verr)
    @warn "No real Millan22 uncertainty for $(RGI_ID); using a 10.0 m/yr placeholder. Regenerate its gridded_data.nc with add_error=true to fix this."
    10.0
else
    only(verr)
end
@info "λ_V weight" σ_V real=!isnothing(verr)
H_obs = last(series.H)
mask_H = H_obs .!= 0

for mult in MULTIPLIERS
    λ_C = mult * λ_ref
    @info "═══ $(LAW_NAME)  mult = $(mult)  λ_C = $(λ_C) ═══"

    params = build_params(;
        λ_H = 1/σ_H^2, λ_V = 1/σ_V^2, λ_C = λ_C, λ_H₀ = λ_H₀,
        maxC = maxC, tspan = tspan, t₀ = t₀,
        gridScalingFactor = GSF, epochs = EPOCHS,
    )
    model = build_model(params, glaciers, A_VALUE)
    inversion = Inversion(model, glaciers, params)

    elapsed = @elapsed run!(inversion)

    C = inverted_C(inversion)
    # basis = :surface: this is compared against Millan22, a surface velocity observation.
    frac = Huginn.sliding_fraction(C, Huginn.inn1(glacier.H₀), A_VALUE;
        p = LAW.p, q = LAW.q, basis = :surface)
    # Every C statistic is taken over cells that carry ice. Over the full grid they are
    # meaningless: most staggered cells are ice free, contribute to no loss term, and keep
    # their seed forever (`sliding_fraction` already returns NaN there). `roughness`/`reg_C`
    # stay unmasked, since the loss computes them over the whole grid. See `ice_mask_C` in
    # common.jl.
    msk = ice_mask_C(glacier)
    Cm = C[msk]
    fracm = frac[msk]

    res = inversion.results.simulation[1]
    v_rmse = sqrt(mean((res.V[end][mask_V] .- V_ref[mask_V]) .^ 2))
    h_rmse = sqrt(mean((res.H[end][mask_H] .- H_obs[mask_H]) .^ 2))
    rough = mean(ODINN.∇²(C, glacier.Δx, glacier.Δy) .^ 2)

    # The regularization term as the loss actually computes it, not as `roughness` reports it.
    # `SlidingRegularization` evaluates `sum((∇²C)²)` over an all-true mask, once, at
    # `t == tspan[1]` (Regularization.jl:340-364 and the TikhonovRegularization body at :147).
    # `roughness` above is an unmasked *mean*, so the two differ by the cell count -- which is
    # precisely the factor that moves under grid refinement, and therefore the thing that must
    # not be guessed at.
    #
    # `reg_frac` is the resolution-robust quantity. A multiplier calibrated at one grid does
    # not transfer: λ_ref ∝ Δx⁴/ncells ∝ Δx⁶, while for a C with a fixed physical correlation
    # length the penalty scales as ncells ∝ Δx⁻², so the effective regularization weakens as
    # Δx⁴ under refinement -- about 81x going from gsf 3 to gsf 1. Match `reg_frac` across
    # resolutions rather than `mult`, and read `reg_frac ≈ 0` as the regularization being off
    # regardless of how large λ_C looks.
    reg_C = λ_C * sum(ODINN.∇²(C, glacier.Δx, glacier.Δy) .^ 2)
    loss_total = inversion.results.stats.losses[end]
    reg_frac = reg_C / max(abs(loss_total), eps())

    push!(results, (
        String(LAW_NAME), mult, λ_C, loss_total,
        mean(Cm), median(Cm), maximum(Cm), mean(Cm .> 0.95 * maxC),
        rough, reg_C, reg_frac, median(fracm), v_rmse, h_rmse, elapsed,
    ))
    C_fields[mult] = C
    results_by_mult[mult] = res

    @info "Done" mult C_mean=mean(Cm) at_bound=mean(Cm .> 0.95*maxC) slide_frac=round(median(fracm); digits=3) v_rmse=round(v_rmse; digits=2) reg_frac seconds=round(elapsed; digits=1)

    # Self-contained per-multiplier subfolder: safe to write even from a process that only
    # ever handles this one multiplier (the parallel case), since its path is unique to `mult`.
    mult_dir = run_dir("02_lambda_sweep", TAG, "mult$(mult)")
    CSV.write(joinpath(mult_dir, "summary.csv"), results[end:end, :])
    jldsave(joinpath(mult_dir, "fields.jld2");
        C, res, glacier, A = A_VALUE, A_name = A_NAME, mult, λ_C, maxC, law = LAW)

    # Aggregate across every multiplier this process has run so far -- only safe to write
    # here when this process owns the whole sweep (see CAMPAIGN_DIR's comment above).
    if length(MULTIPLIERS) > 1
        CSV.write(joinpath(CAMPAIGN_DIR, "summary.csv"), results)
        jldsave(joinpath(run_dir("02_lambda_sweep", TAG, "fields"), "C_fields.jld2");
            C_fields, results_by_mult, glacier, A = A_VALUE, A_name = A_NAME,
            multipliers = MULTIPLIERS, λ_ref, maxC, law = LAW)
    end

    f = Figure(size = (1300, 450))
    ax1 = Axis(f[1, 1], title = "C", aspect = DataAspect())
    hm1 = heatmap!(ax1, C; colormap = :viridis)
    Colorbar(f[1, 1], hm1; vertical = true, halign = :right, tellwidth = false)
    ax2 = Axis(f[1, 2], title = "sliding fraction", aspect = DataAspect())
    hm2 = heatmap!(ax2, frac; colormap = :magma, colorrange = (0, 1))
    Colorbar(f[1, 2], hm2; vertical = true, halign = :right, tellwidth = false)
    ax3 = Axis(f[1, 3], title = "V modelled − observed  (m/yr)", aspect = DataAspect())
    dV = zeros(size(V_ref)); dV[mask_V] = res.V[end][mask_V] .- V_ref[mask_V]
    hm3 = heatmap!(ax3, dV; colormap = :RdBu)
    Colorbar(f[1, 3], hm3; vertical = true, halign = :right, tellwidth = false)
    Label(f[0, :],
        "$(LAW_NAME)  A=$(A_NAME)  p=$(LAW.p) q=$(LAW.q)   λ_C = $(mult)·λ_ref   |   V RMSE = $(round(v_rmse; digits=2)) m/yr";
        fontsize = 17, font = :bold)
    save(joinpath(mult_dir, "plot.pdf"), f)
end

# Aggregate figures: only this process's own multipliers are in `C_fields`/`results`, so
# only draw them here when this process ran the whole sweep. After a parallel run (one
# multiplier per process) use `02_lambda_sweep_merge.jl` to build these from every
# `mult*/` subfolder instead.
if length(MULTIPLIERS) > 1
    # One shared colorrange/colorbar across panels, not auto-scaled per panel, so panels are
    # actually comparable to each other.
    present_mults = filter(m -> haskey(C_fields, m), MULTIPLIERS)
    crange = extrema(reduce(vcat, vec.(getindex.(Ref(C_fields), present_mults))))
    fig = Figure(size = (380 * length(present_mults) + 120, 460))
    for (k, mult) in enumerate(present_mults)
        ax = Axis(fig[1, k], title = "λ_C = $(mult)·λ_ref", aspect = DataAspect())
        heatmap!(ax, C_fields[mult]; colormap = :viridis, colorrange = crange)
    end
    cticks = range(crange[1], crange[2]; length = 5)
    Colorbar(fig[1, length(present_mults) + 1]; colormap = :viridis, colorrange = crange,
        label = "C", ticks = cticks, tickformat = vs -> [(@sprintf "%.3e" v) for v in vs])
    Label(fig[0, :],
        "Inverted C vs regularization — $(RGI_ID), A=$(A_NAME), $(LAW_NAME) (p=$(LAW.p), q=$(LAW.q))";
        fontsize = 20, font = :bold)
    save(joinpath(CAMPAIGN_DIR, "comparison.pdf"), fig)

    # reg_frac, not roughness: roughness is an unmasked mean that does not transfer across
    # resolutions (see the comment where reg_frac is computed above), while reg_frac is
    # exactly the regularization's share of the loss the optimizer saw, at any grid.
    fig2 = Figure(size = (900, 500))
    ax = Axis(fig2[1, 1], xlabel = "reg_frac  (regularization / total loss)", ylabel = "V RMSE  (m/yr)",
        xscale = log10, title = "L-curve — $(LAW_NAME)")
    scatterlines!(ax, max.(results.reg_frac, eps()), results.v_rmse; markersize = 14)
    for r in eachrow(results)
        text!(ax, max(r.reg_frac, eps()), r.v_rmse; text = " ×$(r.mult)", fontsize = 11)
    end
    save(joinpath(CAMPAIGN_DIR, "lcurve.pdf"), fig2)
end

println(results)
