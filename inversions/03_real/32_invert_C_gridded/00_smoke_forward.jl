"""
Phase 0 — forward smoke test for the gridded C inversion campaign.

Runs Aletsch forward over the full 2009-2018 observation window with mass balance active,
before any inversion machinery is involved. Three things are being checked here:

  1. that the continuous-RHS mass balance branch integrates stably over nine years,
  2. that the date aware glathida ingestion returns the expected survey campaigns,
  3. how long a single forward solve takes, since every adjoint epoch pays that cost.

Nothing is inverted. Run with:
    julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/00_smoke_forward.jl
"""

using Vendr
using CairoMakie
using Printf
using Statistics

const RGI_ID = "RGI60-11.01450"   # Aletsch
const OUT_DIR = joinpath(@__DIR__, "outputs", "00_smoke_forward")
mkpath(OUT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

rgi_paths = get_rgi_paths()

params = Parameters(
    simulation = SimulationParameters(
        use_MB = true,
        step_MB = 1.0/12.0,
        tspan = (2009.0, 2018.0),
        multiprocessing = false,
        workers = 1,
        use_glathida_data = true,
        gridScalingFactor = 1,
        rgi_paths = rgi_paths,
        ice_thickness_source = :Millan22,
        velocity_product = :Millan22,
    ),
    physical = PhysicalParameters(
        maxC = 2e-14,   # see notes: the 8e-17 default caps sliding at ~1% of deformation
    ),
    solver = Huginn.SolverParameters(step = 1.0/12.0),
)

# ─────────────────────────────────────────────────────────────────────────────
# Glacier and observations
# ─────────────────────────────────────────────────────────────────────────────

@info "Initializing glacier" rgi_id = RGI_ID
t_init = @elapsed glaciers = initialize_glaciers([RGI_ID], params)
glacier = only(glaciers)
@info "Glacier ready" elapsed_s=round(t_init; digits = 1) grid=size(glacier.H₀) Δx=glacier.Δx

# Date aware glathida campaigns (Sleipnir feature/glathida-transient)
thickness_series = glathida_thickness_series(glacier, params)
for (t, H) in zip(thickness_series.t, thickness_series.H)
    @info "Glathida campaign" t=round(t; digits = 4) n_cells=count(H .!= 0) mean_H=round(mean(H[H .!= 0]); digits = 1)
end

# ─────────────────────────────────────────────────────────────────────────────
# Model: temperate ice, no sliding, calibrated temperature-index mass balance
# ─────────────────────────────────────────────────────────────────────────────

model = Model(
    iceflow = SIA2Dmodel(params; A = Huginn.TemperateA()),
    mass_balance = TImodel1(params),
    regressors = (;),
)

@info "Calibrating mass balance against Hugonnet geodetic MB"
t_calib = @elapsed model = calibrate_MB_model(model, glaciers, params)
@info "Mass balance calibrated" elapsed_s=round(t_calib; digits = 1)

# ─────────────────────────────────────────────────────────────────────────────
# Forward run
# ─────────────────────────────────────────────────────────────────────────────

prediction = Prediction(model, glaciers, params)
@info "Running forward solve" tspan = params.simulation.tspan
t_solve = @elapsed run!(prediction)
@info "Forward solve done" elapsed_s=round(t_solve; digits = 1)

result = prediction.results[1]
H₀, H₁ = glacier.H₀, result.H[end]

vol₀ = sum(H₀) * glacier.Δx * glacier.Δy / 1e9
vol₁ = sum(H₁) * glacier.Δx * glacier.Δy / 1e9

@info "Volume change" km3_start=round(vol₀; digits = 3) km3_end=round(vol₁; digits = 3) percent=round(100*(vol₁-vol₀)/vol₀; digits = 2)

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic figure
# ─────────────────────────────────────────────────────────────────────────────

fig = Figure(size = (1400, 420))
for (k, (field, title, cmap)) in enumerate((
    (H₀, "H at $(params.simulation.tspan[1])  (m)", :Blues),
    (H₁, "H at $(params.simulation.tspan[2])  (m)", :Blues),
    (H₁ .- H₀, "ΔH over the run  (m)", :RdBu),
))
    ax = Axis(fig[1, k], title = title, aspect = DataAspect())
    hm = heatmap!(ax, field; colormap = cmap)
    Colorbar(fig[1, k], hm; vertical = true, halign = :right, tellwidth = false)
end
Label(fig[0, :], "Phase 0 — forward smoke test, $(RGI_ID), A = temperate, C = 0";
    fontsize = 18, font = :bold)

pdf_path = joinpath(OUT_DIR, "forward.pdf")
save(pdf_path, fig)
@info "Figure written" pdf_path

# ─────────────────────────────────────────────────────────────────────────────
# Timing summary — every adjoint epoch pays the forward cost at least twice
# ─────────────────────────────────────────────────────────────────────────────

open(joinpath(OUT_DIR, "timings.csv"), "w") do io
    println(io, "stage,seconds")
    println(io, "initialize_glaciers,$(round(t_init; digits = 2))")
    println(io, "calibrate_MB_model,$(round(t_calib; digits = 2))")
    println(io, "forward_solve,$(round(t_solve; digits = 2))")
end

@printf("\n%-24s %8.1f s\n", "initialize_glaciers", t_init)
@printf("%-24s %8.1f s\n", "calibrate_MB_model", t_calib)
@printf("%-24s %8.1f s\n", "forward_solve", t_solve)
