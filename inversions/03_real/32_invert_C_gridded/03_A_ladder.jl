"""
Phase 3 — gridded C inversion across a ladder of fixed A.

A is pinned and the gridded C absorbs whatever deformation alone cannot explain, descending
from the Cuffey & Paterson temperate value through the range used in the literature. The
regularization multiplier is chosen from the Phase 2 sweep and passed in.

Run with, defaulting to weertman and a multiplier of 1:
    julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/03_A_ladder.jl [law] [mult]
"""

include(joinpath(@__DIR__, "common.jl"))

using CSV
using DataFrames
using JLD2

mkpath(OUT_DIR)

const LAW_NAME = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :weertman
const LAW = getproperty(SLIDING_LAWS, LAW_NAME)
const MULT = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
# Resolution and epoch budget as arguments, as in `02_lambda_sweep.jl`. Full resolution costs
# roughly 24x the forward solve of gsf 3 (outputs/00b_perf_profile.csv), so the ladder is run
# coarse first and at production settings once λ_C is settled.
const GSF = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
const EPOCHS = length(ARGS) >= 4 ? parse.(Int, split(ARGS[4], ",")) : [10, 20]
@assert length(EPOCHS)==2 "epochs must match the two optimizers in build_params, e.g. 10,20"
# Tagged so an exploratory run cannot overwrite a production one.
const TAG = "$(LAW_NAME)_gsf$(GSF)_ep$(join(EPOCHS, "-"))_mult$(MULT)"

const σ_H = 18.0
const σ_V = 10.0
const H_REF = 300.0

# Fixed creep parameters, Pa⁻³ yr⁻¹, read straight off the same fitted polynomial
# A_TEMPERATE comes from (Huginn.polyA_PatersonCuffey(), Laws.jl), rather than hand-transcribed
# decimals: 0 to -20 °C spans 7.57e-17 down to 3.79e-18, about 20x, all real Cuffey & Paterson
# (2010) table entries rather than round numbers that happen to fall between rows.
const A_POLY = Huginn.polyA_PatersonCuffey()
const A_LADDER = [
    ("temperate", A_TEMPERATE),   # Cuffey & Paterson at 0 °C   -- matches A_POLY(0.0)
    ("m5C", A_POLY(-5.0)),        # Cuffey & Paterson at -5 °C
    ("m10C", A_POLY(-10.0)),      # Cuffey & Paterson at -10 °C
    ("m15C", A_POLY(-15.0)),      # Cuffey & Paterson at -15 °C
    ("m20C", A_POLY(-20.0)),      # Cuffey & Paterson at -20 °C
]

C_scale = sliding_scale(LAW, A_TEMPERATE, H_REF)
maxC = 8 * C_scale

probe = build_params(; tspan = (2009.0, 2018.0), t₀ = 2009.0, maxC = maxC,
    gridScalingFactor = GSF)
glacier, series = prepare_glacier(probe)

tspan = aligned_tspan(first(series.t), 2018.0, 1.0/12.0)
t₀ = first(tspan)
@assert all(tspan[1] .<= series.t .<= tspan[2]) "A glathida campaign falls outside the simulation window"

glacier = Sleipnir.Glacier2D(glacier; thicknessData = series)
glaciers = set_sliding_law!([glacier], LAW, 0.1 * maxC)

ncells = prod(size(glacier.H₀) .- 1)
λ_C = MULT * reference_λ(C_scale, glacier.Δx, ncells)

@info "Ladder setup" LAW_NAME maxC λ_C mult=MULT tspan

results = DataFrame(
    law = String[], name = String[], A = Float64[], loss = Float64[],
    C_mean = Float64[], C_median = Float64[], C_max = Float64[], frac_at_bound = Float64[],
    reg_C = Float64[], reg_frac = Float64[],
    slide_frac_mean = Float64[], slide_frac_median = Float64[],
    v_rmse = Float64[], h_rmse = Float64[], seconds = Float64[],
)

C_fields = Dict{String, Matrix{Float64}}()

V_ref = only(glacier.velocityData.vabs)
mask_V = V_ref .> 0.0
H_obs = last(series.H)
mask_H = H_obs .!= 0

for (name, A_value) in A_LADDER
    @info "═══ $(LAW_NAME)  A = $(A_value)  ($(name)) ═══"

    # maxC is kept fixed across the ladder so the C fields stay directly comparable, even
    # though the sliding/deformation crossover moves as A changes
    params = build_params(;
        λ_H = 1/σ_H^2, λ_V = 1/σ_V^2, λ_C = λ_C, λ_H₀ = 1e-4,
        maxC = maxC, tspan = tspan, t₀ = t₀,
        gridScalingFactor = GSF, epochs = EPOCHS,
    )
    model = build_model(params, glaciers, A_value)
    inversion = Inversion(model, glaciers, params)

    elapsed = @elapsed run!(inversion)

    C = inverted_C(inversion)
    frac = sliding_fraction(C, Huginn.inn1(glacier.H₀), A_value)
    # Every C statistic is taken over cells that carry ice. Over the full grid they are
    # meaningless: most staggered cells are ice free, contribute to no loss term, keep their
    # seed forever, and make `sliding_fraction` return exactly 1 because deformation is 0
    # there. `roughness`/`reg_C` stay unmasked, since the loss computes them over the whole
    # grid. See `ice_mask_C` in common.jl.
    msk = ice_mask_C(glacier)
    Cm = C[msk]
    fracm = frac[msk]

    # The regularization term as the loss computes it (`sum((∇²C)²)` over an all-true mask,
    # once, at t = tspan[1]), not as a mean. `reg_frac` is what transfers across resolutions:
    # a multiplier calibrated at one grid does not, since λ_ref ∝ Δx⁶ while the penalty for a
    # C with fixed physical correlation length scales as Δx⁻². Match `reg_frac`, not `mult`.
    reg_C = λ_C * sum(ODINN.∇²(C, glacier.Δx, glacier.Δy) .^ 2)
    loss_total = inversion.results.stats.losses[end]
    reg_frac = reg_C / max(abs(loss_total), eps())

    res = inversion.results.simulation[1]
    v_rmse = sqrt(mean((res.V[end][mask_V] .- V_ref[mask_V]) .^ 2))
    h_rmse = sqrt(mean((res.H[end][mask_H] .- H_obs[mask_H]) .^ 2))

    push!(results, (
        String(LAW_NAME), name, A_value, loss_total,
        mean(Cm), median(Cm), maximum(Cm), mean(Cm .> 0.95 * maxC),
        reg_C, reg_frac,
        mean(fracm), median(fracm), v_rmse, h_rmse, elapsed,
    ))
    C_fields[name] = C

    @info "Result" A=A_value C_mean=mean(Cm) at_bound=mean(Cm .> 0.95*maxC) slide_frac=round(median(fracm); digits=3) v_rmse=round(v_rmse; digits=2) seconds=round(elapsed; digits=1)

    CSV.write(joinpath(OUT_DIR, "03_A_ladder_$(TAG).csv"), results)
    jldsave(joinpath(OUT_DIR, "03_A_ladder_fields_$(TAG).jld2");
        C_fields, ladder = A_LADDER, λ_C, maxC, law = LAW)

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
        "$(LAW_NAME)   A = $(round(A_value*1e17; digits=2))e-17 ($(name))   |   V RMSE = $(round(v_rmse; digits=2)) m/yr";
        fontsize = 17, font = :bold)
    save(joinpath(OUT_DIR, "03_A_ladder_$(TAG)_$(name).pdf"), f)
end

fig = Figure(size = (400 * length(A_LADDER), 460))
for (k, (name, A_value)) in enumerate(A_LADDER)
    haskey(C_fields, name) || continue
    ax = Axis(fig[1, k],
        title = "$(name)\nA = $(round(A_value*1e17; digits=2))e-17", aspect = DataAspect())
    hm = heatmap!(ax, C_fields[name]; colormap = :viridis)
    Colorbar(fig[1, k], hm; vertical = true, halign = :right, tellwidth = false)
end
Label(fig[0, :], "Inverted C across the A ladder — $(RGI_ID), $(LAW_NAME)";
    fontsize = 20, font = :bold)
save(joinpath(OUT_DIR, "03_A_ladder_$(TAG)_comparison.pdf"), fig)

fig2 = Figure(size = (900, 500))
ax = Axis(fig2[1, 1], xlabel = "A  (Pa⁻³ yr⁻¹)", ylabel = "median sliding fraction",
    xscale = log10, title = "Share of SIA diffusivity carried by sliding — $(LAW_NAME)")
scatterlines!(ax, results.A, results.slide_frac_median; markersize = 14)
save(joinpath(OUT_DIR, "03_A_ladder_$(TAG)_sliding_fraction.pdf"), fig2)

println(results)
