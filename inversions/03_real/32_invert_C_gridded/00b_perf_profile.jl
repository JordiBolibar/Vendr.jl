"""
Phase 0b — where the forward solve spends its time.

A 9 year solve takes ~60 s on the native grid, which puts a full A ladder out of reach in
one night. This measures how the cost splits between mass balance and ice flow, how it
scales with the run length, and what coarsening the grid buys, before any of it is traded
away.

Run with:
    julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/00b_perf_profile.jl
"""

using Vendr
using Printf
using Statistics

const RGI_ID = "RGI60-11.01450"
const OUT_DIR = joinpath(@__DIR__, "outputs")
mkpath(OUT_DIR)

function make_params(; use_MB, tspan, gsf, step = 1.0/12.0)
    Parameters(
        simulation = SimulationParameters(
            use_MB = use_MB,
            step_MB = 1.0/12.0,
            tspan = tspan,
            multiprocessing = false,
            workers = 1,
            use_glathida_data = false,
            gridScalingFactor = gsf,
            rgi_paths = get_rgi_paths(),
            ice_thickness_source = :Millan22,
            velocity_product = :Millan22,
        ),
        physical = PhysicalParameters(maxC = 2e-14),
        solver = Huginn.SolverParameters(step = step),
    )
end

function time_solve(glaciers, params)
    model = Model(
        iceflow = SIA2Dmodel(params; A = Huginn.TemperateA()),
        mass_balance = params.simulation.use_MB ? TImodel1(params) : nothing,
        regressors = (;),
    )
    if params.simulation.use_MB
        model = calibrate_MB_model(model, glaciers, params)
    end
    prediction = Prediction(model, glaciers, params)
    run!(prediction)                      # warm up, so compilation is not measured
    return @elapsed run!(Prediction(model, glaciers, params))
end

rows = Tuple{String, Int, Float64}[]

for gsf in (1, 2, 3)
    # Initialized with use_MB so the raw climate covers the Hugonnet window. The period is
    # fixed at initialize_glaciers time, so glaciers built without it cannot calibrate later.
    p_init = make_params(; use_MB = true, tspan = (2009.0, 2018.0), gsf = gsf)
    t_init = @elapsed glaciers = initialize_glaciers([RGI_ID], p_init)
    n = prod(size(only(glaciers).H₀))
    @info "Grid" gsf size=size(only(glaciers).H₀) cells=n init_s=round(t_init; digits = 1)

    for (label, use_MB, tspan) in (
        ("9yr_noMB", false, (2009.0, 2018.0)),
        ("9yr_MB", true, (2009.0, 2018.0)),
        ("1yr_MB", true, (2017.0, 2018.0)),
    )
        params = make_params(; use_MB = use_MB, tspan = tspan, gsf = gsf)
        t = time_solve(glaciers, params)
        push!(rows, ("gsf$(gsf)_$(label)", n, t))
        @printf("  %-18s cells=%6d  %7.2f s\n", label, n, t)
    end
end

# Solver step size, which sets how often the mass balance source term is evaluated
p_init = make_params(; use_MB = true, tspan = (2009.0, 2018.0), gsf = 1)
glaciers = initialize_glaciers([RGI_ID], p_init)
n = prod(size(only(glaciers).H₀))
for step in (1.0/12.0, 1.0/4.0, 1.0)
    params = make_params(; use_MB = true, tspan = (2009.0, 2018.0), gsf = 1, step = step)
    t = time_solve(glaciers, params)
    push!(rows, ("gsf1_9yr_MB_step$(round(step; digits = 3))", n, t))
    @printf("  step=%-8.4f       cells=%6d  %7.2f s\n", step, n, t)
end

open(joinpath(OUT_DIR, "00b_perf_profile.csv"), "w") do io
    println(io, "config,cells,seconds")
    for (label, cells, t) in rows
        println(io, "$(label),$(cells),$(round(t; digits = 3))")
    end
end

println("\n=== summary ===")
for (label, cells, t) in rows
    @printf("%-32s %6d cells  %7.2f s\n", label, cells, t)
end
