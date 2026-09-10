"""
Phase 0c — cost of one adjoint epoch.

The forward solve sets a floor, but the ladder budget is decided by the backward pass. This
times a single Adam epoch of the real gridded C inversion at a few grid resolutions, so the
number of epochs and rungs that fit in a night follows from a measurement rather than a
guess.

Run with:
    julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/00c_adjoint_timing.jl
"""

include(joinpath(@__DIR__, "common.jl"))

using Printf

mkpath(OUT_DIR)

rows = Tuple{Int, Int, Float64, Float64}[]

for gsf in (1, 2, 3)
    probe = build_params(; tspan = (2009.0, 2018.0), t₀ = 2009.0, gridScalingFactor = gsf)
    t_init = @elapsed glacier, series = prepare_glacier(probe)

    t₀ = first(series.t)
    tspan = (t₀, 2018.0)
    glacier = Sleipnir.Glacier2D(glacier; thicknessData = series)
    glaciers = [glacier]
    ncells = prod(size(glacier.H₀))

    # A single Adam epoch, no LBFGS, so the measurement is one forward plus one backward
    params = build_params(;
        λ_H = 1/18.0^2, λ_V = 1/10.0^2, λ_C = 1e30,
        epochs = 1,
        gridScalingFactor = gsf,
        tspan = tspan, t₀ = t₀,
    )
    model = build_model(params, glaciers, A_TEMPERATE)
    inversion = Inversion(model, glaciers, params)

    t_epoch = @elapsed run!(inversion)

    push!(rows, (gsf, ncells, t_init, t_epoch))
    @printf("gsf=%d  cells=%6d  init=%6.1f s  epoch=%7.1f s\n", gsf, ncells, t_init, t_epoch)
end

open(joinpath(OUT_DIR, "00c_adjoint_timing.csv"), "w") do io
    println(io, "gsf,cells,init_s,epoch_s")
    for (gsf, n, ti, te) in rows
        println(io, "$(gsf),$(n),$(round(ti; digits = 2)),$(round(te; digits = 2))")
    end
end

println("\n=== budget at 80 epochs per rung, 6 rungs ===")
for (gsf, n, _, te) in rows
    @printf("gsf=%d : %5.1f min/rung   %5.1f h for the full ladder\n",
        gsf, 80 * te / 60, 6 * 80 * te / 3600)
end
