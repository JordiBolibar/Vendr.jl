"""
Phase 1 — synthetic recovery of a gridded sliding coefficient.

No end-to-end gridded C inversion has ever been run in ODINN: the existing tests only cover
the seeding of `GriddedInv(params, glaciers, :C)`, and `test_grad_finite_diff` has no `:C`
branch at all. So before any real-data result can be interpreted, the pipeline has to be
shown to recover a C field that is known by construction.

Ground truth is generated with Huginn's `SyntheticC` law, which maps cumulative positive
degree days and topographic roughness onto `[minC, maxC]`. The inversion then starts from
the default seed (C ≈ 5e-5·maxC, essentially no sliding) and has to climb back to it.

Run with:
    julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/01_synthetic_C.jl
"""

include(joinpath(@__DIR__, "common.jl"))

const USE_MB = false          # isolate the sliding signal first; Phase 0 covers MB transport
const GSF = 3                 # coarser grid keeps the validation cheap
const TSPAN = (2010.0, 2012.0)
const MAXC = 2e-14

const RUN_DIR = run_dir("01_synthetic_C")

# ─────────────────────────────────────────────────────────────────────────────
# Ground truth with a known, spatially variable C
# ─────────────────────────────────────────────────────────────────────────────

params_gt = Parameters(
    simulation = SimulationParameters(
        use_MB = USE_MB,
        step_MB = 1.0/12.0,
        tspan = TSPAN,
        multiprocessing = false,
        workers = 1,
        use_glathida_data = false,
        gridScalingFactor = GSF,
        rgi_paths = get_rgi_paths(),
        ice_thickness_source = :Millan22,
        velocity_product = :Millan22,
    ),
    physical = PhysicalParameters(maxC = MAXC, minC = 0.05 * MAXC),
    solver = Huginn.SolverParameters(step = 1.0/12.0),
)

@info "Initializing glacier" rgi_id=RGI_ID gsf=GSF
glaciers = initialize_glaciers([RGI_ID], params_gt)
@info "Grid" size=size(only(glaciers).H₀)

model_gt = Model(
    iceflow = SIA2Dmodel(params_gt;
        A = Huginn.ConstantA(A_TEMPERATE),
        C = SyntheticC(params_gt),
    ),
    mass_balance = USE_MB ? TImodel1(params_gt) : nothing,
    regressors = (;),
)

tstops = collect(TSPAN[1]:0.5:TSPAN[2])
@info "Generating synthetic ground truth"
t_gt = @elapsed glaciers_gt = generate_ground_truth(
    glaciers, params_gt, model_gt, tstops; store = (:H, :V))
@info "Ground truth ready" elapsed_s=round(t_gt; digits = 1)

# Evaluate the true C field on the grid so it can be compared to the inverted one
prediction_gt = Prediction(model_gt, glaciers_gt, params_gt)
run!(prediction_gt)
C_true = copy(prediction_gt.cache.iceflow.C)
@info "True C" min=minimum(C_true) max=maximum(C_true) mean=mean(C_true)

# ─────────────────────────────────────────────────────────────────────────────
# Inversion
# ─────────────────────────────────────────────────────────────────────────────

params_inv = Parameters(
    simulation = params_gt.simulation,
    hyper = Hyperparameters(
        batch_size = 1,
        epochs = [15, 40],
        optimizer = [
            ODINN.Adam(0.05),
            ODINN.LBFGS(linesearch = ODINN.LineSearches.BackTracking(iterations = 5)),
        ],
    ),
    physical = params_gt.physical,
    UDE = UDEparameters(
        grad = SciMLSensitivityAdjoint(),
        optim_autoAD = ODINN.Optimization.AutoZygote(),
        empirical_loss_function = MultiLoss(
            losses = (LossH(loss = L2Sum(distance = 0)), LossV()),
            λs = (1.0, 1.0),
        ),
    ),
    solver = Huginn.SolverParameters(step = 1.0/12.0),
)

model_inv = Model(
    iceflow = SIA2Dmodel(params_inv;
        A = Huginn.ConstantA(A_TEMPERATE),
        C = LawC(params_inv; scalar = false),
    ),
    mass_balance = USE_MB ? TImodel1(params_inv) : nothing,
    regressors = (; C = GriddedInv(params_inv, glaciers_gt, :C),),
)

inversion = Inversion(model_inv, glaciers_gt, params_inv)
@info "Running gridded C inversion"
t_inv = @elapsed run!(inversion)
@info "Inversion done" elapsed_s=round(t_inv; digits = 1)

C_inv = inverted_C(inversion)

rel_err = abs.(C_inv .- C_true) ./ max.(C_true, eps())
@info "Recovery" mean_rel_err=round(100*mean(rel_err); digits = 2) median_rel_err=round(100*median(rel_err); digits = 2)

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

fig = Figure(size = (1500, 450))
scale = 1e15   # plot C in 1e-15 units so the colourbars stay readable
for (k, (field, title)) in enumerate((
    (C_true .* scale, "C ground truth (SyntheticC)"),
    (C_inv .* scale, "C inverted"),
    ((C_inv .- C_true) .* scale, "difference"),
))
    ax = Axis(fig[1, k], title = title, aspect = DataAspect())
    hm = heatmap!(ax, field; colormap = k == 3 ? :RdBu : :viridis)
    Colorbar(fig[1, k], hm; vertical = true, halign = :right, tellwidth = false)
end
Label(fig[0, :],
    "Phase 1 — synthetic gridded C recovery, $(RGI_ID), " *
    "mean rel. err = $(round(100*mean(rel_err); digits = 1))%";
    fontsize = 18, font = :bold)

save(joinpath(RUN_DIR, "synthetic_C.pdf"), fig)
@info "Figure written" path=joinpath(RUN_DIR, "synthetic_C.pdf")

open(joinpath(RUN_DIR, "timings.csv"), "w") do io
    println(io, "stage,seconds")
    println(io, "ground_truth,$(round(t_gt; digits = 2))")
    println(io, "inversion,$(round(t_inv; digits = 2))")
end
