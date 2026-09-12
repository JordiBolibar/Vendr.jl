"""
Phase 4 — sliding coefficient implied pointwise by the SIA and the observations.

No inversion is involved. In the SIA the surface velocity is `V = D‖∇S‖` with

    D = C(p-q+2)(ρg)^(p-q) H̄^(p-q+1) ∇S^(n-1) + (2A(ρg)ⁿ/(n+1)) H̄^(n+1) ∇S^(n-1)

so with A fixed and H, ∇S and V all observed, C follows algebraically at every pixel. This
answers the question the campaign was built for, directly and in seconds: given a physically
realistic A, what sliding does the SIA require to match Millan velocities, and is it
realistic?

Where the implied C is negative the SIA already moves ice too fast at that A before any
sliding is added, which is the mechanism behind the below-range A of Lisa Girod's thesis,
here resolved in space rather than inferred from a scalar.

Caveats: this is a diagnostic, not the inversion. It uses the Millan thickness as truth
rather than solving for H₀, applies no spatial regularization, and is instantaneous rather
than transient. It brackets what any inversion should find.

Run with:
    julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/04_implied_C.jl
"""

include(joinpath(@__DIR__, "common.jl"))

using CSV
using DataFrames
using Printf

mkpath(OUT_DIR)

const A_LADDER = [
    ("temperate", A_TEMPERATE),   # Cuffey & Paterson (2010) at 0 °C
    ("millan22", 4.0e-17),
    ("girod_mean", 2.97e-17),     # mean of Lisa's inverted A field
    ("A_2e-17", 2.0e-17),
    ("A_1e-17", 1.0e-17),
    ("A_5e-18", 5.0e-18),
]

params = build_params(;
    tspan = (2009.0, 2018.0), t₀ = 2009.0, maxC = 1.0, gridScalingFactor = 1)
glaciers = initialize_glaciers([RGI_ID], params)
glacier = only(glaciers)

(; ρ, g) = params.physical
ρg = ρ * g
n = glacier.n

# Staggered fields, matching what surface_V computes internally
S = glacier.B .+ glacier.H₀
dSdx = Huginn.diff_x(S) / glacier.Δx
dSdy = Huginn.diff_y(S) / glacier.Δy
∇S = (Huginn.avg_y(dSdx) .^ 2 .+ Huginn.avg_x(dSdy) .^ 2) .^ 0.5
H̄ = Huginn.avg(glacier.H₀)

V_obs = Huginn.inn1(only(glacier.velocityData.vabs))
@assert size(V_obs) == size(∇S) "observation grid $(size(V_obs)) does not match staggered grid $(size(∇S))"

# Only where the SIA is meaningful and the observation exists
mask = (V_obs .> 0) .& (H̄ .> 10.0) .& (∇S .> 1e-3)
@info "Domain" cells=count(mask) mean_H=round(mean(H̄[mask]); digits=1) mean_V=round(mean(V_obs[mask]); digits=1) mean_slope_deg=round(rad2deg(atan(mean(∇S[mask]))); digits=1)

D_obs = V_obs ./ ∇S

results = DataFrame(
    law = String[], A_name = String[], A = Float64[],
    frac_negative = Float64[], C_median = Float64[], C_p90 = Float64[],
    slide_frac_median = Float64[],
)

fields = Dict{Tuple{String, String}, Matrix{Float64}}()

for (law_name, law) in pairs(SLIDING_LAWS), (A_name, A) in A_LADDER
    deform = @. (2 * A * ρg^n / (n + 1)) * H̄^(n + 1) * ∇S^(n - 1)
    slide_unit = @. (law.p - law.q + 2) * ρg^(law.p - law.q) *
                    H̄^(law.p - law.q + 1) * ∇S^(n - 1)
    C_implied = (D_obs .- deform) ./ slide_unit

    vals = C_implied[mask]
    frac_neg = mean(vals .< 0)
    pos = vals[vals .> 0]

    # Share of the diffusivity carried by sliding, where sliding is physically possible
    sf = isempty(pos) ? 0.0 :
         median((slide_unit[mask][vals .> 0] .* pos) ./ D_obs[mask][vals .> 0])

    push!(results, (String(law_name), A_name, A, frac_neg,
        isempty(pos) ? NaN : median(pos), isempty(pos) ? NaN : quantile(pos, 0.9), sf))

    masked = fill(NaN, size(C_implied))
    masked[mask] = C_implied[mask]
    fields[(String(law_name), A_name)] = masked

    @printf("%-9s %-11s A=%.2e : %5.1f%% negative, median C+ = %.3e, sliding share = %.2f\n",
        law_name, A_name, A, 100 * frac_neg,
        isempty(pos) ? NaN : median(pos), sf)
end

CSV.write(joinpath(OUT_DIR, "04_implied_C.csv"), results)

# One panel per A, per law: where is sliding possible at all, and how much
for law_name in ("weertman", "budd")
    fig = Figure(size = (380 * length(A_LADDER), 460))
    for (k, (A_name, A)) in enumerate(A_LADDER)
        F = fields[(law_name, A_name)]
        pos = filter(!isnan, F); pos = pos[pos .> 0]
        hi = isempty(pos) ? 1.0 : quantile(pos, 0.95)
        ax = Axis(fig[1, k],
            title = "$(A_name)\nA = $(round(A*1e17; digits=2))e-17\n" *
                    "$(round(100*results[(results.law .== law_name) .& (results.A_name .== A_name), :frac_negative][1]; digits=1))% negative",
            aspect = DataAspect())
        hm = heatmap!(ax, F; colormap = :RdBu, colorrange = (-hi, hi))
        Colorbar(fig[1, k], hm; vertical = true, halign = :right, tellwidth = false)
    end
    Label(fig[0, :],
        "Sliding coefficient implied by SIA + Millan22 observations — $(RGI_ID), $(law_name)" *
        "  (red = negative, SIA already too fast)";
        fontsize = 19, font = :bold)
    save(joinpath(OUT_DIR, "04_implied_C_$(law_name).pdf"), fig)
end

# The headline curve: how much of the glacier cannot accommodate any sliding
fig = Figure(size = (900, 520))
ax = Axis(fig[1, 1], xlabel = "A  (Pa⁻³ yr⁻¹)", xscale = log10,
    ylabel = "fraction of glacier with negative implied C",
    title = "Where the SIA already overpredicts velocity before any sliding")
for law_name in ("weertman", "budd")
    sub = results[results.law .== law_name, :]
    scatterlines!(ax, sub.A, sub.frac_negative; markersize = 13, label = law_name)
end
axislegend(ax; position = :rt)
save(joinpath(OUT_DIR, "04_implied_C_negative_fraction.pdf"), fig)

println()
println(results)
