"""
Campaign inversion on Aletsch with the chosen settings, at any resolution.

Gridded C and free H₀, mass balance on. Loss: H on the glathida surveys, the Millan22 velocity
averaged over its window, and the regularization of H₀ and C. The optimizers are Adam and then
the LBFGS of `ODINN.default_optimizer()`, run one after the other. The outputs are written after
each optimizer, so a late failure of a long run keeps the results of the first one:

  - `summary.csv`: final loss, H RMSE per glathida campaign, V RMSE over the Millan22 window,
    C statistics, change of H₀ from Millan22, and time,
  - `losses.csv` and `losses.pdf`: the loss at each iteration,
  - `fields/C_fields.jld2`: C, the results and the glacier, with the layout of
    `02_lambda_sweep.jl`, so that `05_diagnostics.jl` draws the maps (it labels the run
    `mult=<multiplier of λ_C>`).

Settings chosen on Aletsch at gridScalingFactor 4 (see `06_tune_lr.jl`): Adam lr 0.05, λ_H₀ 1e-4
(it makes no difference between 1e-4 and 1), and A at the -10 °C rung of the A ladder.

The weight of the regularization of C does not transfer between resolutions: for the same C
field the penalty grows like Δx⁻², while `reference_λ` shrinks like Δx⁶. The multiplier of
`reference_λ` is therefore (4/gsf)⁴ by default, which gives the same regularization as at
gridScalingFactor 4 (for example 256 at gridScalingFactor 1).

Run with, defaults gsf 4, epochs 40,60, lr 0.05, A rung m10C, λ_H₀ 1e-4:
    julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/07_run_inversion.jl [gsf] [epochs] [lr] [A_name] [λ_H₀] [mult]
"""

include(joinpath(@__DIR__, "common.jl"))

using CSV
using DataFrames
using JLD2

const GSF = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
const EPOCHS = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) : [40, 60]
const LR = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.05
const A_NAME = length(ARGS) >= 4 ? ARGS[4] : "m10C"
const A_VALUE = A_from_name(A_NAME)
const λ_H₀ = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1e-4
const MULT_DEFAULT = (4 / GSF)^4
const MULT = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : MULT_DEFAULT
const LAW = SLIDING_LAWS.weertman
const σ_H = 18.0
const σ_V = 10.0
const H_REF = 300.0
const TAG = "weertman_$(A_NAME)_gsf$(GSF)_ep$(join(EPOCHS, "-"))_lr$(LR)" *
            (λ_H₀ == 1e-4 ? "" : "_lH0$(λ_H₀)") * (MULT == MULT_DEFAULT ? "" : "_mult$(MULT)")
const DIR = run_dir("07_run_inversion", TAG)

C_scale = Huginn.sliding_scale(A_VALUE, H_REF; p = LAW.p, q = LAW.q)
maxC = 8 * C_scale

probe = build_params(; tspan = (2009.0, 2018.0), t₀ = 2009.0, maxC = maxC,
    gridScalingFactor = GSF)
glacier, series = prepare_glacier(probe)
tspan = aligned_tspan(first(series.t), 2018.0, 1.0/12.0)
t₀ = first(tspan)
glacier = Sleipnir.Glacier2D(glacier; thicknessData = series)
glaciers = set_sliding_law!([glacier], LAW, 0.1 * maxC)
λ_C = MULT * reference_λ(C_scale, glacier.Δx, prod(size(glacier.H₀) .- 1))

V_ref = only(glacier.velocityData.vabs)
mask_V = V_ref .> 0.0
t1_V = Sleipnir.datetime_to_floatyear(only(glacier.velocityData.date1))
t2_V = Sleipnir.datetime_to_floatyear(only(glacier.velocityData.date2))
ice = glacier.H₀ .> 0

# Same metrics as `01c_e2e_campaign_run.jl` and `06_tune_lr.jl`
function fit(res)
    h = map(series.t, series.H) do t, H_obs
        k = argmin(abs.(res.t .- t))
        m = H_obs .!= 0
        sqrt(mean((res.H[k][m] .- H_obs[m]) .^ 2))
    end
    window = findall(t1_V .<= res.t .<= t2_V)
    V̄ = mean(res.V[window])
    v = sqrt(mean((V̄[mask_V] .- V_ref[mask_V]) .^ 2))
    return (; h, v)
end

"""
Write the outputs of the run from the current state of `inversion`.
"""
function write_outputs(inversion, fit₀, elapsed)
    losses = inversion.results.stats.losses
    fit₁ = fit(only(inversion.results.simulation))
    C = inverted_C(inversion)
    Cm = C[ice_mask_C(glacier)]
    H₀ = inversion.results.stats.initial_conditions[String(glacier.rgi_id)]

    summary = DataFrame(
        gsf = GSF, lr = LR, mult = MULT, loss_start = losses[begin],
        loss_adam = losses[EPOCHS[1]], loss_end = losses[end], iterations = length(losses),
        h_rmse_start = join(round.(fit₀.h; digits = 2), ";"),
        h_rmse_end = join(round.(fit₁.h; digits = 2), ";"),
        v_rmse_start = fit₀.v, v_rmse_end = fit₁.v,
        C_median = median(Cm), C_near_max = mean(Cm .> 0.95 * maxC),
        C_near_zero = mean(Cm .< 0.05 * maxC),
        ΔH₀_mean = mean(abs.(H₀[ice] .- glacier.H₀[ice])),
        ΔH₀_max = maximum(abs.(H₀[ice] .- glacier.H₀[ice])),
        seconds = elapsed)
    CSV.write(joinpath(DIR, "summary.csv"), summary)
    CSV.write(joinpath(DIR, "losses.csv"), DataFrame(iteration = eachindex(losses), loss = losses))

    jldsave(out_path(joinpath(DIR, "fields"), "C_fields.jld2");
        C_fields = Dict(MULT => C), results_by_mult = Dict(MULT => inversion.results.simulation[1]),
        glacier, A = A_VALUE, A_name = A_NAME, maxC, law = LAW, H₀)

    f = Figure(size = (700, 400))
    ax = Axis(f[1, 1]; xlabel = "iteration", ylabel = "loss", yscale = log10,
        title = "gsf $(GSF), lr = $(LR): Adam, then LBFGS")
    lines!(ax, eachindex(losses), losses)
    vlines!(ax, [EPOCHS[1]]; color = :gray, linestyle = :dash)
    save(joinpath(DIR, "losses.pdf"), f)

    @info "Written" DIR loss_end=losses[end] h_rmse=fit₁.h v_rmse=fit₁.v seconds=round(elapsed)
end

@info "Setup" GSF EPOCHS LR A_NAME λ_H₀ MULT grid=size(glacier.H₀) tspan DIR
optimizers = [ODINN.Adam(LR), last(ODINN.default_optimizer())]
params = build_params(;
    λ_H = 1/σ_H^2, λ_V = 1/σ_V^2, λ_C = λ_C, λ_H₀ = λ_H₀,
    maxC = maxC, tspan = tspan, t₀ = t₀, gridScalingFactor = GSF, epochs = EPOCHS,
    optimizer = optimizers)
inversion = Inversion(build_model(params, glaciers, A_VALUE), glaciers, params)

fit₀ = fit(only(ODINN.create_results(inversion.model.trainable_components.θ, inversion,
    map; processVelocity = Huginn.V_from_H)))
# One optimizer at a time: each `run!` starts from the parameters the previous one left
elapsed = 0.0
for (optimizer, epochs) in zip(optimizers, EPOCHS)
    inversion.parameters.hyper.optimizer = optimizer
    inversion.parameters.hyper.epochs = epochs
    global elapsed += @elapsed run!(inversion; path_tb_logger = nothing)
    write_outputs(inversion, fit₀, elapsed)
end
