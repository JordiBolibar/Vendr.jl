"""
Gradient check of the full campaign loss against finite differences.

`01_validate_C_gradient.jl` checks `LossAvgV` alone, without mass balance and with an adaptive
solver. This script checks the loss the campaign really uses: `LossH` on the glathida surveys,
`LossAvgV` on Millan22, the regularization of H₀ and of C, with mass balance, over the full
window. Both θ.C and θ.IC are checked.

Finite differences need a fixed time step (`adaptive = false`) and `supply_eigen_est = true`:
otherwise ROCK changes its number of stages between two close values of θ, and the
finite differences measure that noise instead of a derivative.

Run with:
    julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/01b_validate_campaign_gradient.jl [A_name]
"""

include(joinpath(@__DIR__, "common.jl"))

using Random
using LinearAlgebra

const GSF = 4               # coarse grid: each FD component costs a full forward solve
const N_FD = 4              # number of components checked, for θ.C and for θ.IC
const FD_DELTAS = [1e-4, 1e-5]
const LAW = SLIDING_LAWS.weertman
# A colder rung, so that C has some work to do (see `02_lambda_sweep.jl`)
const A_NAME = isempty(ARGS) ? "m10C" : ARGS[1]
const A_VALUE = A_from_name(A_NAME)
const σ_H = 18.0
const σ_V = 10.0
const H_REF = 300.0

Random.seed!(1234)

C_scale = Huginn.sliding_scale(A_VALUE, H_REF; p = LAW.p, q = LAW.q)
maxC = 8 * C_scale

"""
Same as `build_params`, but with a fixed step, our eigenvalue estimate for ROCK2, and
`use_glathida_data = false`: with it, `initialize_glaciers` calls Sleipnir's
`get_glathida_glacier`, which puts the full resolution glathida indices into the coarsened grid
and fails for `gridScalingFactor > 1`. The glathida data reach the loss through
`glathida_thickness_series` anyway, which does coarsen them.
"""
function check_params(tspan, t₀, λ_C)
    return Parameters(
        simulation = SimulationParameters(
            use_MB = true,
            step_MB = 1.0/12.0,
            tspan = tspan,
            multiprocessing = false,
            workers = 1,
            use_glathida_data = false,
            gridScalingFactor = GSF,
            rgi_paths = get_rgi_paths(),
            ice_thickness_source = :Millan22,
            velocity_product = :Millan22,
        ),
        hyper = Hyperparameters(batch_size = 1, epochs = 1, optimizer = ODINN.Adam(0.01)),
        physical = PhysicalParameters(maxC = maxC),
        UDE = UDEparameters(
            grad = SciMLSensitivityAdjoint(),
            optim_autoAD = ODINN.Optimization.AutoZygote(),
            empirical_loss_function = MultiLoss(
                losses = (LossH(loss = L2Sum(distance = 0)), LossAvgV(),
                    InitialThicknessRegularization(t₀), SlidingRegularization()),
                λs = (1/σ_H^2, 1/σ_V^2, 1e-4, λ_C)),
            initial_condition_filter = :Zang1980,
        ),
        solver = Huginn.SolverParameters(
            step = 1.0/12.0, solver = Huginn.ROCK2(), adaptive = false, dt = 1.0/120.0,
            supply_eigen_est = true),
    )
end

glacier, series = prepare_glacier(check_params((2009.0, 2018.0), 2009.0, 0.0))
tspan = aligned_tspan(first(series.t), 2018.0, 1.0/12.0)
t₀ = first(tspan)
glacier = Sleipnir.Glacier2D(glacier; thicknessData = series)
glaciers = set_sliding_law!([glacier], LAW, 0.1 * maxC)
λ_C = reference_λ(C_scale, glacier.Δx, prod(size(glacier.H₀) .- 1))
params = check_params(tspan, t₀, λ_C)

model = build_model(params, glaciers, A_VALUE)
inversion = Inversion(model, glaciers, params)
θ = inversion.model.trainable_components.θ
@info "Setup" A_NAME tspan grid=size(glacier.H₀) campaigns=series.t n_C=length(θ.C) n_IC=length(θ.IC)

t_adj = @elapsed g = ODINN.grad_loss_iceflow!(copy(θ), inversion, map)
@info "Adjoint gradient" seconds=round(t_adj; digits = 1) norm_C=norm(g.C) norm_IC=norm(g.IC)
@assert all(isfinite, g) "The adjoint gradient has non-finite entries"

loss(θv) = ODINN.loss_iceflow_transient(θv, inversion, map)

# `getproperty` gives a view; `θ.C[Symbol("1")]` would be a copy and perturb nothing
component(p, block::Symbol) = vec(getproperty(getproperty(p, block), Symbol("1")))

function perturbed(θ, block::Symbol, k::Integer, δ::Float64)
    θp = copy(θ)
    component(θp, block)[k] += δ
    return θp
end

# Largest adjoint sensitivities on thick ice, where the signal is above round-off
function pick(block::Symbol, thick)
    gb = component(g, block)
    candidates = findall(vec(thick) .& isfinite.(gb))
    @assert length(candidates) >= N_FD "Fewer than $(N_FD) thick cells for θ.$(block)"
    return candidates[sortperm(abs.(gb[candidates]); rev = true)[1:N_FD]]
end

L0 = loss(copy(θ))
@info "Base loss" L0

rel_errs = Dict(:C => Float64[], :IC => Float64[])
@printf("\n%-4s %-8s %16s %16s %11s %16s %11s\n",
    "", "cell", "adjoint", "fd(1e-4)", "rel.err", "fd(1e-5)", "rel.err")
for (block, thick) in ((:C, Huginn.avg(glacier.H₀) .> 100.0), (:IC, glacier.H₀ .> 100.0))
    for k in pick(block, thick)
        ad = component(g, block)[k]
        fds = map(FD_DELTAS) do δ
            (loss(perturbed(θ, block, k, δ)) - loss(perturbed(θ, block, k, -δ))) / (2 * δ)
        end
        rels = [abs(fd - ad) / max(abs(fd), abs(ad), eps()) for fd in fds]
        append!(rel_errs[block], rels)
        @printf("%-4s %-8d %16.6e %16.6e %11.3e %16.6e %11.3e\n",
            block, k, ad, fds[1], rels[1], fds[2], rels[2])
        flush(stdout)
    end
end

for block in (:C, :IC)
    m = median(rel_errs[block])
    @printf("\nθ.%s median relative error: %.3e\n", block, m)
    if m < 1e-3
        @info "PASS θ.$(block)"
    else
        @error "FAIL θ.$(block): the adjoint disagrees with finite differences" median_rel_err=m
    end
end
