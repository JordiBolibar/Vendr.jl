"""
End-to-end check of the campaign inversion: does `run!` work with the full setup?

`01b_validate_campaign_gradient.jl` shows the gradient is correct at the starting point. This
script runs the inversion itself, with the production settings of `build_params` and
`build_model`: gridded C and free H₀, `LossH` on the glathida surveys, `LossAvgV` on Millan22,
regularization of H₀ and C, mass balance, adaptive ROCK2, Adam then LBFGS.

It passes when the run ends without error and:
  - the loss is finite and goes down,
  - the fit to the H and V observations is better than at the start,
  - most of C is not stuck at a bound of the tanh (a stuck parameter looks converged),
  - the inverted H₀ is finite and not negative.

Run with, defaults weertman, gsf 4, epochs 10,10, A rung m10C:
    julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/01c_e2e_campaign_run.jl [gsf] [epochs] [A_name]
"""

include(joinpath(@__DIR__, "common.jl"))

using Test

const LAW = SLIDING_LAWS.weertman
const GSF = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
const EPOCHS = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) : [10, 10]
const A_NAME = length(ARGS) >= 3 ? ARGS[3] : "m10C"
const A_VALUE = A_from_name(A_NAME)
# Multiplies λ_H; 0 keeps only the velocity data, to check that the V term alone improves V
const H_WEIGHT = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
const σ_H = 18.0
const σ_V = 10.0
const H_REF = 300.0

# Comparison of LBFGS setups. LBFGS_INIT: first step length of the line search, `default`
# (α = 1 on -g, which can blow up the forward when the loss is large), `scaled` (step norm
# capped to 1 in θ) or `hagerzhang` (sized from ‖θ‖/‖g‖). LBFGS_LS: `backtracking` (as in
# `build_params`) or `hagerzhang`. LOSS_SCALE: `sigma` (λ = 1/σ²) or `unit` (λ = 1, a large loss).
const LBFGS_INIT = get(ENV, "LBFGS_INIT", "default")
const LBFGS_LS = get(ENV, "LBFGS_LS", "backtracking")
const UNIT_WEIGHTS = get(ENV, "LOSS_SCALE", "sigma") == "unit"
alphaguess = LBFGS_INIT == "scaled" ?
             ODINN.LineSearches.InitialStatic(alpha = 1.0, scaled = true) :
             # α0 = NaN, or the first step is not sized from ‖θ‖/‖g‖
             LBFGS_INIT == "hagerzhang" ? ODINN.LineSearches.InitialHagerZhang(α0 = NaN) :
             ODINN.LineSearches.InitialStatic()
linesearch = LBFGS_LS == "hagerzhang" ? ODINN.LineSearches.HagerZhang() :
             ODINN.LineSearches.BackTracking(iterations = 5)
optimizer = [ODINN.Adam(adam_lr(GSF)), ODINN.LBFGS(; alphaguess, linesearch)]

C_scale = Huginn.sliding_scale(A_VALUE, H_REF; p = LAW.p, q = LAW.q)
maxC = 8 * C_scale

probe = build_params(; tspan = (2009.0, 2018.0), t₀ = 2009.0, maxC = maxC,
    gridScalingFactor = GSF)
glacier, series = prepare_glacier(probe)
tspan = aligned_tspan(first(series.t), 2018.0, 1.0/12.0)
t₀ = first(tspan)
glacier = Sleipnir.Glacier2D(glacier; thicknessData = series)
# Diagnostic: the Millan22 loader keeps vy positive to the north, while the model's +y goes
# along the matrix index, which points south on a north-up grid
if get(ENV, "FLIP_VY", "false") == "true"
    only(glacier.velocityData.vy) .*= -1
    @warn "vy of Millan22 negated (FLIP_VY)"
end
glaciers = set_sliding_law!([glacier], LAW, 0.1 * maxC)
λ_C = reference_λ(C_scale, glacier.Δx, prod(size(glacier.H₀) .- 1))

params = build_params(;
    λ_H = UNIT_WEIGHTS ? H_WEIGHT : H_WEIGHT/σ_H^2, λ_V = UNIT_WEIGHTS ? 1.0 : 1/σ_V^2,
    λ_C = λ_C, λ_H₀ = 1e-4, maxC = maxC, tspan = tspan, t₀ = t₀, gridScalingFactor = GSF,
    epochs = EPOCHS, optimizer = optimizer)
model = build_model(params, glaciers, A_VALUE)
inversion = Inversion(model, glaciers, params)
@info "Setup" A_NAME GSF EPOCHS tspan grid=size(glacier.H₀) campaigns=series.t LBFGS_INIT LBFGS_LS UNIT_WEIGHTS

V_ref = only(glacier.velocityData.vabs)
mask_V = V_ref .> 0.0

# Millan22 window, the one `LossAvgV` averages the modelled velocity over
t1_V = Sleipnir.datetime_to_floatyear(only(glacier.velocityData.date1))
t2_V = Sleipnir.datetime_to_floatyear(only(glacier.velocityData.date2))

"""
RMSE of the modelled H at each glathida campaign (nearest saved time), and of the modelled V
averaged over the Millan22 window, which is what `LossAvgV` fits.
"""
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

θ₀ = copy(inversion.model.trainable_components.θ)
fit₀ = fit(only(ODINN.create_results(
    θ₀, inversion, map; processVelocity = Huginn.V_from_H)))
@info "Fit at the start" h_rmse=fit₀.h v_rmse=fit₀.v

# No TensorBoard log: it would be written inside the ODINN folder
elapsed = @elapsed run!(inversion; path_tb_logger = nothing)

losses = inversion.results.stats.losses
fit₁ = fit(only(inversion.results.simulation))
C = inverted_C(inversion)
Cm = C[ice_mask_C(glacier)]
# Near zero means no sliding, which is a valid answer. Near maxC means stuck at the cap.
at_upper = mean(Cm .> 0.95 * maxC)
at_lower = mean(Cm .< 0.05 * maxC)
H₀ = inversion.results.stats.initial_conditions[String(glacier.rgi_id)]

@printf("\niterations = %d   loss: %.4e -> %.4e (ratio %.3e)   %.0f s\n",
    length(losses), losses[begin], losses[end], losses[end] / losses[begin], elapsed)
println("loss history: ", join(round.(losses; sigdigits = 5), ", "))
for (k, t) in enumerate(series.t)
    @printf("H RMSE at %.2f: %.2f -> %.2f m\n", t, fit₀.h[k], fit₁.h[k])
end
@printf("V RMSE: %.2f -> %.2f m/yr\n", fit₀.v, fit₁.v)
@printf("C over ice: median %.3e, share near maxC %.3f, share near 0 %.3f (maxC = %.3e)\n",
    median(Cm), at_upper, at_lower, maxC)
@printf("Millan22 window: %.2f to %.2f\n", t1_V, t2_V)
@printf("H₀: min %.2f, max %.2f m\n", minimum(H₀), maximum(H₀))

@testset "Campaign inversion end to end" begin
    @test all(isfinite, losses)
    @test losses[end] < losses[begin]
    H_WEIGHT > 0 && @test sum(fit₁.h) < sum(fit₀.h)
    @test fit₁.v < fit₀.v
    @test at_upper < 0.5
    @test all(isfinite, H₀) && all(H₀ .>= 0)
end
