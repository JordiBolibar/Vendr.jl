"""
Shared setup for the gridded basal sliding inversion on Aletsch.

H₀ and the gridded C are both free. The run starts at the first glathida campaign so that
survey pins H₀, while the second campaign and the Millan22 velocities constrain the
trajectory, and hence C.

Gradients come from the automatic adjoint: LawC defines no p_VJP!, so manual adjoints would
silently return a zero gradient for θ.C.
"""

using Vendr
using CairoMakie
using Printf
using Statistics
using Dates

const RGI_ID = "RGI60-11.01450"   # Aletsch
const OUT_DIR = joinpath(@__DIR__, "outputs")

# Cuffey & Paterson (2010) temperate ice, in Pa⁻³ yr⁻¹. This is `Huginn.TemperateA()`.
const A_TEMPERATE = Huginn.polyA_PatersonCuffey()(0.0)

# Sliding laws, as (p, q) of u_b = C τ_b^p / N^q. Weertman is the primary here: Gilbert
# et al. (2023) constrain m = 3.1 ± 0.3 on hard bedded Argentière and find Weertman
# suitable for long term evolution. Budd is kept as a sensitivity check.
const SLIDING_LAWS = (
    weertman = (p = 3.0, q = 0.0),
    budd = (p = 3.0, q = 2.0),
)

"""
    sliding_scale(law, A, H; ρg = 900 * 9.81, n = 3)

Value of `C` at which sliding and deformation contribute equally to the SIA diffusivity.

Huginn splits the diffusivity as `C(ρg)^(p-q) H^(p-q+1) ∇S^(p-1)` against
`2A(ρg)ⁿHⁿ⁺²∇Sⁿ⁻¹/(n+2)`, so the crossover depends on the sliding law and sets both the
bound on `C` and the scale of its regularization. The `∇S` powers cancel when `p-1 = n-1`.
"""
function sliding_scale(law, A::Float64, H::Float64; ρg::Float64 = 900.0*9.81, n::Float64 = 3.0)
    deformation = 2 * A * ρg^n * H^(n + 2) / (n + 2)
    sliding_unit_C = ρg^(law.p - law.q) * H^(law.p - law.q + 1)
    return deformation / sliding_unit_C
end

"""
    reference_λ(scale, Δx, ncells; target = 0.1)

Weight making a Tikhonov term worth `target` of a unit data misfit at the given field scale.

The penalty is `λ Σ(∇²x)²` and `∇²x` scales as `x/Δx²`, so expressing λ this way keeps the
sweep comparable between sliding laws whose `C` differ by orders of magnitude.
"""
function reference_λ(scale::Float64, Δx::Float64, ncells::Integer; target::Float64 = 0.1)
    return target / (ncells * (scale / Δx^2)^2)
end

"""
    aligned_tspan(t_first_obs, t_end, step_MB)

Simulation window ending at `t_end` and starting at or before `t_first_obs`.

The mass balance requires `tspan` to be a whole number of `step_MB`, so the start cannot be
placed on the survey date itself. Rounding up would push the first campaign outside the
window and silently drop it, hence the start is snapped to the step at or below it.
"""
function aligned_tspan(t_first_obs::Float64, t_end::Float64, step_MB::Float64)
    nsteps = ceil((t_end - t_first_obs) / step_MB)
    return (t_end - nsteps * step_MB, t_end)
end

"""
    set_sliding_law!(glaciers, law, C0)

Set the sliding exponents of each glacier, which the default `p` and `q` laws read, and seed
`glacier.C`, which `GriddedInv` fills the whole gridded θ.C field from.

`C0` is required rather than defaulted. `glacier.C` defaults to 0.0 in Sleipnir, and C = 0
maps to the GriddedInv seed θ = -5, which is also the flattest point of the tanh
parameterization: `dC/dθ = maxC*sech²(5)/2`, about 7e-5 of its maximum. Starting there, the
loss is nearly flat in θ across the whole grid and the optimizer barely moves -- on the
Aletsch campaign the loss moved 0.5% in 15 iterations and LBFGS's line search gave up after
about 5, regardless of its epoch budget, while every reported C statistic sat at the seed
value. Nothing errors, so it looks converged rather than broken.

`C0` should be inside (0, maxC); 0.1*maxC is the value the gradient gate
(`01_validate_C_gradient.jl`) validates against.
"""
function set_sliding_law!(glaciers, law, C0::Float64)
    for glacier in glaciers
        glacier.p = law.p
        glacier.q = law.q
        glacier.C = C0
    end
    return glaciers
end

"""
    build_params(; λs..., epochs, maxC, gridScalingFactor, tspan, t₀)

Build the `Parameters` of one scenario.

`maxC` has to be passed explicitly because its scale depends on the sliding law, see
[`sliding_scale`](@ref). The Sleipnir default of 8e-17 is only meaningful for a Weertman
law and is many orders off for the default Budd one.
"""
function build_params(;
        λ_H::Float64 = 1.0,
        λ_V::Float64 = 1.0,
        λ_C::Float64 = 0.0,
        λ_H₀::Float64 = 1e-4,
        epochs = [10, 20],
        optimizer = [
            ODINN.Adam(0.05),
            ODINN.LBFGS(linesearch = ODINN.LineSearches.BackTracking(iterations = 5)),
        ],
        maxC::Float64 = 2e-14,
        gridScalingFactor::Int = 1,
        tspan::Tuple{Float64, Float64},
        t₀::Float64,
        solver = Huginn.ROCK2(),
        adaptive::Bool = true,
        dt::Float64 = 1.0/120.0,
        abstol::Float64 = 1e-3,
)
    losses = Any[
        LossH(loss = L2Sum(distance = 0)),   # glathida is sparse: any erosion empties the mask
        LossAvgV(),
        InitialThicknessRegularization(t₀),
    ]
    λs = Any[λ_H, λ_V, λ_H₀]

    if λ_C > 0.0
        push!(losses, SlidingRegularization())
        push!(λs, λ_C)
    end

    return Parameters(
        simulation = SimulationParameters(
            use_MB = true,
            step_MB = 1.0/12.0,
            tspan = tspan,
            multiprocessing = false,
            workers = 1,
            use_glathida_data = true,
            gridScalingFactor = gridScalingFactor,
            rgi_paths = get_rgi_paths(),
            ice_thickness_source = :Millan22,
            velocity_product = :Millan22,
        ),
        hyper = Hyperparameters(
            batch_size = 1,
            epochs = epochs,
            optimizer = optimizer,
        ),
        physical = PhysicalParameters(maxC = maxC),
        UDE = UDEparameters(
            grad = SciMLSensitivityAdjoint(),
            optim_autoAD = ODINN.Optimization.AutoZygote(),
            empirical_loss_function = MultiLoss(
                losses = Tuple(losses), λs = Tuple(λs)),
            initial_condition_filter = :Zang1980,
        ),
        # `InterpolatingAdjoint` is not stable in backward mode with the SIA, so the solver
        # has to be an explicit stabilized one. `SolverParameters` defaults to `RDPK3Sp35`,
        # which is not, and `with_eigen_est` only rewrites ROCK2/ROCK4, so leaving the
        # default silently runs an unstabilized method on both passes and the gradient comes
        # back wrong with a healthy looking norm. Gradient checks additionally need
        # `adaptive = false`: adaptive stepping makes the solution discontinuous in θ and the
        # finite differences then measure step acceptance jitter instead of a derivative.
        # `abstol` is the accuracy the gradient inherits: at the 1e-3 default the direction is
        # perturbed a couple of degrees, which is fine for an optimiser but too loose to check
        # an adjoint against finite differences. Tighten it for gradient checks.
        solver = Huginn.SolverParameters(
            step = 1.0/12.0, solver = solver, adaptive = adaptive, dt = dt,
            abstol = abstol),
    )
end

"""
    prepare_glacier(params_probe) -> (glacier, campaigns)

Initialize Aletsch and read its GlaThiDa survey campaigns.

A probe `Parameters` is needed because the campaign dates are only known after the raw
measurements are read, and they in turn define the simulation `tspan`.
"""
function prepare_glacier(params_probe)
    glaciers = initialize_glaciers([RGI_ID], params_probe)
    glacier = only(glaciers)
    series = glathida_thickness_series(glacier, params_probe)
    return glacier, series
end

"""
    build_model(params, glaciers, A_value)

Build the inversion model: fixed scalar A, gridded trainable C, free H₀.

`target` is left at its default since only the manual adjoints use it.
"""
function build_model(params, glaciers, A_value::Float64; mass_balance = TImodel1(params))
    iceflow = SIA2Dmodel(params;
        A = Huginn.ConstantA(A_value),
        C = LawC(params; scalar = false),
    )
    return Model(
        iceflow = iceflow,
        mass_balance = mass_balance,
        regressors = (;
            C = GriddedInv(params, glaciers, :C),
            IC = InitialCondition(params, glaciers, :Millan22),
        ),
    )
end

"""
    inverted_C(inversion, glacier_idx = 1)

Map the trained parameters of one glacier back to physical C values.
"""
function inverted_C(inversion, glacier_idx::Integer = 1)
    θ = inversion.results.stats.θ
    max_C = inversion.parameters.physical.maxC
    return @. max_C * (tanh(θ.C[Symbol("$(glacier_idx)")]) + 1) / 2
end

"""
    ice_mask_C(glacier)

Staggered mask of cells that carry ice, matching the shape of the inverted `C`.

Every C statistic has to be taken over this, not over the whole grid. More than half the
staggered cells are ice free: they contribute to no loss term, so their gradient is exactly
zero and they keep their seed forever. Averaged in, they make `C_mean`/`C_median` report the
seed value rather than anything the inversion did, and they make `sliding_fraction` return
exactly 1 (deformation is `2AH/(n+2)`, which is 0 when `H` is 0, so the ratio is `C/C`).
That combination produced a confident and completely wrong reading of the first sweep:
"the median cell is 100% sliding". Where there is ice the fraction was ~4e-4.
"""
ice_mask_C(glacier) = Huginn.inn1(glacier.H₀) .> 0.0

"""
    sliding_fraction(C, H, A, n = 3)

Share of the SIA diffusivity due to sliding rather than deformation.

In `D = (C + 2AH/(n+2))(ρg)ⁿHⁿ⁺¹‖∇S‖ⁿ⁻¹` the split only depends on `C` against `2AH/(n+2)`,
so this tells whether a recovered C matters dynamically or is just non-zero.
"""
function sliding_fraction(C, H, A::Float64; n::Float64 = 3.0)
    deformation = @. 2 * A * H / (n + 2)
    return @. C / (C + deformation)
end
