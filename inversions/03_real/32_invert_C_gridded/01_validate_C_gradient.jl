"""
Validation gate — adjoint gradient of a velocity loss with respect to a gridded C.

The velocity half of the automatic adjoint had never been exercised: every
SciMLSensitivityAdjoint test in ODINN uses a thickness loss, and the V_from_H rrule that
makes velocity losses differentiable lives on an unmerged branch whose Huginn counterpart
was never pushed. Both halves are ported into this environment, so nothing downstream should
be trusted until the adjoint is checked against finite differences.

Finite differences need a non adaptive solver: adaptive stepping makes the solution a
discontinuous function of θ, so a perturbation can flip which steps are accepted and the
difference quotient stops measuring a derivative.

Run with:
    julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/01_validate_C_gradient.jl
"""

include(joinpath(@__DIR__, "common.jl"))

using Printf
using Random
using LinearAlgebra

const GSF = 4               # coarse grid: each FD component costs a full forward solve
# Must cover the Millan22 velocity window: LossAvgV reads t1,t2 from velocityData.date1/date2
# (2017-01-01 to 2018-01-01) and averages the prediction over it. A tspan that does not
# contain that window makes indFromT return nothing and silently poisons the gradient.
const TSPAN = (2017.0, 2018.0)
const N_FD = 6              # number of θ.C components checked
# Two step sizes in θ space, where C = maxC(tanh(θ)+1)/2. Too small measures cancellation
# rather than a derivative, so agreement between the two is what makes the check credible.
const FD_DELTAS = [1e-4, 1e-5]

Random.seed!(1234)

law = SLIDING_LAWS.weertman
C_scale = sliding_scale(law, A_TEMPERATE, 300.0)
maxC = 8 * C_scale

params = Parameters(
    simulation = SimulationParameters(
        use_MB = false,
        tspan = TSPAN,
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
        empirical_loss_function = LossAvgV(),
    ),
    # Adaptive stepping makes the solution a discontinuous function of θ, which normally
    # invalidates finite differences. It is kept here because a fixed step run leaves
    # tstops missing from solution.t and indFromT then returns nothing, and because the
    # discrepancy being tested spans tens of orders of magnitude, far above step noise.
    # ROCK2, explicitly. `SolverParameters` defaults to `RDPK3Sp35`, an unstabilized explicit
    # method, and `with_eigen_est` only rewrites ROCK2/ROCK4 — so leaving the solver unnamed
    # ran an unstabilized method on a stiff parabolic problem in both directions. That is what
    # produced the NaN adjoint here (backward solve aborting at t=2009.406, step error 9.8e9),
    # and elsewhere a wrong-sign gradient with a healthy-looking norm and no NaNs. See 20583a0.
    solver = Huginn.SolverParameters(step = 1.0/12.0, solver = Huginn.ROCK2()),
)

glaciers = initialize_glaciers([RGI_ID], params)
set_sliding_law!(glaciers, law, 0.1 * maxC)
glacier = only(glaciers)
@info "Setup" grid=size(glacier.H₀) maxC C_scale tspan=TSPAN

# No mass balance: with use_MB=false the raw climate only spans tspan, so calibrating a
# TImodel1 inside the Inversion constructor would fail on the Hugonnet window
model = build_model(params, glaciers, A_TEMPERATE; mass_balance = nothing)
inversion = Inversion(model, glaciers, params)
θ = inversion.model.trainable_components.θ

@info "Parameters" n_θ=length(θ) n_C=length(θ.C)

# ── adjoint gradient ─────────────────────────────────────────────────────────
t_adj = @elapsed g_adj = ODINN.grad_loss_iceflow!(copy(θ), inversion, map)
gC = g_adj.C[Symbol("1")]
@info "Adjoint gradient" seconds=round(t_adj; digits=1) norm=norm(gC) nonzero=count(!iszero, gC)

if all(iszero, gC)
    @error "Adjoint gradient wrt θ.C is identically zero — the velocity VJP is not reaching C"
    exit(1)
end
if any(!isfinite, gC)
    @error "Adjoint gradient contains non-finite entries" n_nan=count(isnan, gC) n_inf=count(isinf, gC) n=length(gC)
end

# ── finite differences on a few components ───────────────────────────────────
loss(θv) = ODINN.loss_iceflow_transient(θv, inversion, map)

"""
    perturbed(θ, k, δ)

Copy of `θ` with the `k`-th component of the gridded C shifted by `δ`.

`getproperty` is deliberate: ComponentArrays returns a copy for `getindex(::Symbol)` and a
view only for `getproperty`, so `θ.C[Symbol("1")][k] += δ` silently perturbs nothing and
every finite difference comes out exactly zero.
"""
function perturbed(θ, k::Integer, δ::Float64)
    θp = copy(θ)
    getproperty(θp.C, Symbol("1"))[k] += δ
    return θp
end

L0 = loss(copy(θ))
@info "Base loss" L0

# Pick components with the largest adjoint sensitivity, where the signal beats round-off.
# Restricted to thick, finite cells: `sortperm(...; rev=true)` sorts NaN first, so a single
# bad entry would otherwise fill the whole sample with ice-free cells whose FD is exactly 0.
H̄ = Huginn.avg(glacier.H₀)
candidates = findall(vec(isfinite.(gC) .& (H̄ .> 100.0)))
@assert length(candidates) >= N_FD "Fewer than $(N_FD) thick cells with a finite gradient"
idx = candidates[sortperm(abs.(vec(gC)[candidates]); rev = true)[1:N_FD]]

@printf("\n%-8s %16s %16s %11s %16s %11s\n",
    "cell", "adjoint", "fd(1e-4)", "rel.err", "fd(1e-5)", "rel.err")
rel_errs = Float64[]
for k in idx
    ad = vec(gC)[k]
    fds = map(FD_DELTAS) do δ
        (loss(perturbed(θ, k, δ)) - loss(perturbed(θ, k, -δ))) / (2 * δ)
    end
    rels = [abs(fd - ad) / max(abs(fd), abs(ad), eps()) for fd in fds]
    append!(rel_errs, rels)
    @printf("%-8d %16.6e %16.6e %11.3e %16.6e %11.3e\n",
        k, ad, fds[1], rels[1], fds[2], rels[2])
end

@printf("\nmedian relative error: %.3e\n", median(rel_errs))
if median(rel_errs) < 1e-3
    @info "PASS — adjoint agrees with finite differences, the ported velocity VJP is trustworthy"
else
    @error "FAIL — adjoint disagrees with finite differences, do not trust downstream results" median_rel_err=median(rel_errs)
end
