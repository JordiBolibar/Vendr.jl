"""
Phase 2 — regularization sweep for the gridded C inversion on Aletsch.

A is fixed at the Cuffey & Paterson temperate value and only λ_C varies, so the smoothness
of the recovered C can be picked visually before running the A ladder. λ_C is expressed as
a multiple of a reference weight computed from the C scale of the sliding law in use, which
keeps the sweep comparable between laws whose C differ by orders of magnitude.

Each λ writes its PDF and appends to the CSV as soon as it finishes, so partial results are
usable while the sweep runs.

Run with, law defaulting to weertman:
    julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/02_lambda_sweep.jl [weertman|budd]
"""

include(joinpath(@__DIR__, "common.jl"))

using CSV
using DataFrames
using JLD2

mkpath(OUT_DIR)

const LAW_NAME = isempty(ARGS) ? :weertman : Symbol(ARGS[1])
const LAW = getproperty(SLIDING_LAWS, LAW_NAME)

const σ_H = 18.0    # m, from the glathida repeat surveys
const σ_V = 10.0    # m/yr, representative Millan22 uncertainty
const H_REF = 300.0 # m, representative trunk thickness used to set the C and H₀ scales

# Spans four orders of magnitude: λ_ref fixes the scaling between laws correctly, but its
# absolute value rests on the unknown normalization inside L2Sum, so the sweep is wide.
const MULTIPLIERS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]

# C at which sliding equals deformation, which sets both the bound and the regularization
C_scale = sliding_scale(LAW, A_TEMPERATE, H_REF)
maxC = 8 * C_scale   # headroom to ~89% sliding before the tanh bound bites

@info "Sliding law" LAW_NAME p=LAW.p q=LAW.q C_scale maxC

probe = build_params(; tspan = (2009.0, 2018.0), t₀ = 2009.0, maxC = maxC)
glacier, series = prepare_glacier(probe)

tspan = aligned_tspan(first(series.t), 2018.0, 1.0/12.0)
t₀ = first(tspan)
@info "Window" tspan campaigns=series.t
@assert all(tspan[1] .<= series.t .<= tspan[2]) "A glathida campaign falls outside the simulation window"

glacier = Sleipnir.Glacier2D(glacier; thicknessData = series)
glaciers = set_sliding_law!([glacier], LAW)

ncells = prod(size(glacier.H₀) .- 1)
λ_ref = reference_λ(C_scale, glacier.Δx, ncells)
# Lisa's tuned value, kept fixed so the sweep varies λ_C alone. The computed reference is
# logged for comparison and revisited once λ_C is settled.
λ_H₀ = 1e-4

@info "Weights" λ_ref λ_H₀ λ_H₀_computed=reference_λ(H_REF, glacier.Δx, prod(size(glacier.H₀)); target = 0.05) tspan grid=size(glacier.H₀)

results = DataFrame(
    law = String[], mult = Float64[], λ_C = Float64[], loss = Float64[],
    C_mean = Float64[], C_median = Float64[], C_max = Float64[], frac_at_bound = Float64[],
    roughness = Float64[], slide_frac_median = Float64[],
    v_rmse = Float64[], h_rmse = Float64[], seconds = Float64[],
)

C_fields = Dict{Float64, Matrix{Float64}}()

V_ref = only(glacier.velocityData.vabs)
mask_V = V_ref .> 0.0
H_obs = last(series.H)
mask_H = H_obs .!= 0

for mult in MULTIPLIERS
    λ_C = mult * λ_ref
    @info "═══ $(LAW_NAME)  mult = $(mult)  λ_C = $(λ_C) ═══"

    params = build_params(;
        λ_H = 1/σ_H^2, λ_V = 1/σ_V^2, λ_C = λ_C, λ_H₀ = λ_H₀,
        maxC = maxC, tspan = tspan, t₀ = t₀,
    )
    model = build_model(params, glaciers, A_TEMPERATE)
    inversion = Inversion(model, glaciers, params)

    elapsed = @elapsed run!(inversion)

    C = inverted_C(inversion)
    frac = sliding_fraction(C, Huginn.inn1(glacier.H₀), A_TEMPERATE)

    res = inversion.results.simulation[1]
    v_rmse = sqrt(mean((res.V[end][mask_V] .- V_ref[mask_V]) .^ 2))
    h_rmse = sqrt(mean((res.H[end][mask_H] .- H_obs[mask_H]) .^ 2))
    rough = mean(ODINN.∇²(C, glacier.Δx, glacier.Δy) .^ 2)

    push!(results, (
        String(LAW_NAME), mult, λ_C, inversion.results.stats.loss[end],
        mean(C), median(C), maximum(C), mean(C .> 0.95 * maxC),
        rough, median(frac), v_rmse, h_rmse, elapsed,
    ))
    C_fields[mult] = C

    @info "Done" mult C_mean=mean(C) at_bound=mean(C .> 0.95*maxC) slide_frac=round(median(frac); digits=3) v_rmse=round(v_rmse; digits=2) seconds=round(elapsed; digits=1)

    CSV.write(joinpath(OUT_DIR, "02_lambda_sweep_$(LAW_NAME).csv"), results)
    jldsave(joinpath(OUT_DIR, "02_lambda_fields_$(LAW_NAME).jld2");
        C_fields, multipliers = MULTIPLIERS, λ_ref, maxC, law = LAW)

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
        "$(LAW_NAME)  p=$(LAW.p) q=$(LAW.q)   λ_C = $(mult)·λ_ref   |   V RMSE = $(round(v_rmse; digits=2)) m/yr";
        fontsize = 17, font = :bold)
    save(joinpath(OUT_DIR, "02_lambda_$(LAW_NAME)_mult$(mult).pdf"), f)
end

fig = Figure(size = (400 * length(MULTIPLIERS), 460))
for (k, mult) in enumerate(MULTIPLIERS)
    haskey(C_fields, mult) || continue
    ax = Axis(fig[1, k], title = "λ_C = $(mult)·λ_ref", aspect = DataAspect())
    hm = heatmap!(ax, C_fields[mult]; colormap = :viridis)
    Colorbar(fig[1, k], hm; vertical = true, halign = :right, tellwidth = false)
end
Label(fig[0, :],
    "Inverted C vs regularization — $(RGI_ID), A temperate, $(LAW_NAME) (p=$(LAW.p), q=$(LAW.q))";
    fontsize = 20, font = :bold)
save(joinpath(OUT_DIR, "02_lambda_sweep_$(LAW_NAME)_comparison.pdf"), fig)

fig2 = Figure(size = (900, 500))
ax = Axis(fig2[1, 1], xlabel = "roughness  mean(∇²C)²", ylabel = "V RMSE  (m/yr)",
    xscale = log10, title = "L-curve — $(LAW_NAME)")
scatterlines!(ax, max.(results.roughness, eps()), results.v_rmse; markersize = 14)
for r in eachrow(results)
    text!(ax, max(r.roughness, eps()), r.v_rmse; text = " ×$(r.mult)", fontsize = 11)
end
save(joinpath(OUT_DIR, "02_lambda_sweep_$(LAW_NAME)_lcurve.pdf"), fig2)

println(results)
