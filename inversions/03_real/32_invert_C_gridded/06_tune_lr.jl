"""
Tuning of the Adam learning rate of the campaign, with λ fixed.

θ.IC is now stored scaled (H₀ / H₀_scale), so `adam_lr` no longer means what it meant. Each
learning rate runs Adam and then the LBFGS of `ODINN.default_optimizer()`, and writes:

  - `summary.csv`: final loss, H RMSE per glathida campaign, V RMSE over the Millan22 window,
    C statistics, change of H₀ from Millan22, and time,
  - `losses.csv` and `losses.pdf`: the loss at each iteration, to see how it converges.

One subfolder per learning rate, so several processes can run at the same time.

Run with, defaults lr 0.005,0.01,0.02,0.05, epochs 30,30, gsf 4, A rung m10C, λ_H₀ 1e-4:
    julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/06_tune_lr.jl [lrs] [epochs] [gsf] [A_name] [λ_H₀]
"""

include(joinpath(@__DIR__, "common.jl"))

using CSV
using DataFrames

const LRS = length(ARGS) >= 1 ? parse.(Float64, split(ARGS[1], ",")) : [0.005, 0.01, 0.02, 0.05]
const EPOCHS = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) : [30, 30]
const GSF = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4
const A_NAME = length(ARGS) >= 4 ? ARGS[4] : "m10C"
const A_VALUE = A_from_name(A_NAME)
# Weight of the regularization of H₀
const λ_H₀ = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1e-4
const LAW = SLIDING_LAWS.weertman
const σ_H = 18.0
const σ_V = 10.0
const H_REF = 300.0
const TAG = "weertman_$(A_NAME)_gsf$(GSF)_ep$(join(EPOCHS, "-"))" *
            (λ_H₀ == 1e-4 ? "" : "_lH0$(λ_H₀)")

C_scale = Huginn.sliding_scale(A_VALUE, H_REF; p = LAW.p, q = LAW.q)
maxC = 8 * C_scale

probe = build_params(; tspan = (2009.0, 2018.0), t₀ = 2009.0, maxC = maxC,
    gridScalingFactor = GSF)
glacier, series = prepare_glacier(probe)
tspan = aligned_tspan(first(series.t), 2018.0, 1.0/12.0)
t₀ = first(tspan)
glacier = Sleipnir.Glacier2D(glacier; thicknessData = series)
glaciers = set_sliding_law!([glacier], LAW, 0.1 * maxC)
λ_C = reference_λ(C_scale, glacier.Δx, prod(size(glacier.H₀) .- 1))

V_ref = only(glacier.velocityData.vabs)
mask_V = V_ref .> 0.0
t1_V = Sleipnir.datetime_to_floatyear(only(glacier.velocityData.date1))
t2_V = Sleipnir.datetime_to_floatyear(only(glacier.velocityData.date2))
ice = glacier.H₀ .> 0

# Same metrics as `01c_e2e_campaign_run.jl`
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

for lr in LRS
    dir = run_dir("06_tune_lr", TAG, "lr$(lr)")
    @info "═══ lr = $(lr) ═══" dir
    params = build_params(;
        λ_H = 1/σ_H^2, λ_V = 1/σ_V^2, λ_C = λ_C, λ_H₀ = λ_H₀,
        maxC = maxC, tspan = tspan, t₀ = t₀, gridScalingFactor = GSF, epochs = EPOCHS,
        optimizer = [ODINN.Adam(lr), last(ODINN.default_optimizer())])
    inversion = Inversion(build_model(params, glaciers, A_VALUE), glaciers, params)

    fit₀ = fit(only(ODINN.create_results(inversion.model.trainable_components.θ, inversion,
        map; processVelocity = Huginn.V_from_H)))
    elapsed = @elapsed run!(inversion; path_tb_logger = nothing)

    losses = inversion.results.stats.losses
    fit₁ = fit(only(inversion.results.simulation))
    C = inverted_C(inversion)
    Cm = C[ice_mask_C(glacier)]
    H₀ = inversion.results.stats.initial_conditions[String(glacier.rgi_id)]

    summary = DataFrame(
        lr = lr, loss_start = losses[begin], loss_adam = losses[EPOCHS[1]],
        loss_end = losses[end], iterations = length(losses),
        h_rmse_start = join(round.(fit₀.h; digits = 2), ";"),
        h_rmse_end = join(round.(fit₁.h; digits = 2), ";"),
        v_rmse_start = fit₀.v, v_rmse_end = fit₁.v,
        C_median = median(Cm), C_near_max = mean(Cm .> 0.95 * maxC),
        C_near_zero = mean(Cm .< 0.05 * maxC),
        ΔH₀_mean = mean(abs.(H₀[ice] .- glacier.H₀[ice])),
        ΔH₀_max = maximum(abs.(H₀[ice] .- glacier.H₀[ice])),
        seconds = elapsed)
    CSV.write(joinpath(dir, "summary.csv"), summary)
    CSV.write(joinpath(dir, "losses.csv"), DataFrame(iteration = eachindex(losses), loss = losses))

    f = Figure(size = (700, 400))
    ax = Axis(f[1, 1]; xlabel = "iteration", ylabel = "loss", yscale = log10,
        title = "lr = $(lr): Adam, then LBFGS")
    lines!(ax, eachindex(losses), losses)
    vlines!(ax, [EPOCHS[1]]; color = :gray, linestyle = :dash)
    save(joinpath(dir, "losses.pdf"), f)

    @info "Done" lr loss_end=losses[end] h_rmse=fit₁.h v_rmse=fit₁.v seconds=round(elapsed)
end
