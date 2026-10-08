"""
Phase 5 — diagnostic plots from a saved `02_lambda_sweep.jl` / `03_A_ladder.jl` run.

Reads one `fields/C_fields.jld2`, written alongside the run's `summary.csv`, and regenerates
every plot from it — no inversion is re-run. This is what makes iterating on plot styling
cheap: change a colormap or add a panel and rerun this script in seconds, instead of an hour
of `run!`.

Four plots per saved configuration (multiplier, or A-ladder rung):

  1. `H` modelled vs each glathida survey vs difference       (`plot_glacier_vs_observations`)
  2. `V` modelled vs Millan22 (window mean) vs difference     (`plot_glacier_vs_observations`)
  3. Inverted `C` distribution against the sliding/deformation crossover scale
                                                               (`plot_field_histogram`)
  4. Sliding fraction distribution                            (`plot_field_histogram`)

Plus one summary bar plot across all configurations: median sliding fraction (Lisa's Fig. 6
idiom, generalized from A to whatever axis the saved run swept).

Run with the path to a run's `fields/C_fields.jld2`, e.g.:
    julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/05_diagnostics.jl \
        outputs/03_A_ladder/weertman_gsf3_ep3-5_mult1.0/fields/C_fields.jld2
"""

include(joinpath(@__DIR__, "common.jl"))

using JLD2

@assert !isempty(ARGS) "usage: 05_diagnostics.jl <path/to/fields/C_fields.jld2>"
const FIELDS_PATH = ARGS[1]
# The plots land next to the run they were rendered from, not next to this script, so a
# regeneration after retuning a plot cannot be confused with the run that produced the fields.
const RUN_ROOT = dirname(dirname(FIELDS_PATH))   # .../<script>/<tag>/fields/… -> .../<tag>
const PLOTS_DIR = joinpath(RUN_ROOT, "diagnostics")
mkpath(PLOTS_DIR)

saved = JLD2.load(FIELDS_PATH)
is_ladder = haskey(saved, "results_by_rung")
C_fields = saved["C_fields"]
results_by_key = is_ladder ? saved["results_by_rung"] : saved["results_by_mult"]
glacier = saved["glacier"]
law = saved["law"]
maxC = saved["maxC"]
# Sort keys so every plot iterates in a sensible order: A descending temperature for the
# ladder (already the dict-insertion order, but Dict does not preserve it after a jldsave
# round trip), multipliers ascending for the sweep.
keys_sorted = is_ladder ?
    first.(saved["ladder"]) ∩ collect(keys(C_fields)) :
    sort(collect(keys(C_fields)))

@info "Loaded" path=FIELDS_PATH is_ladder law keys=keys_sorted

label_for(key) = is_ladder ? String(key) : "mult=$(key)"

for key in keys_sorted
    C = C_fields[key]
    res = results_by_key[key]
    tag = replace(label_for(key), "=" => "")
    kdir = joinpath(PLOTS_DIR, tag)
    mkpath(kdir)

    A_value = is_ladder ? only(a for (n, a) in saved["ladder"] if n == key) : saved["A"]
    C_scale = Huginn.sliding_scale(A_value, 300.0; p = law.p, q = law.q)

    fig_H = plot_glacier_vs_observations(res, glacier, :H;
        title = "H — $(label_for(key))")
    save(joinpath(kdir, "H_vs_obs.pdf"), fig_H)

    fig_V = plot_glacier_vs_observations(res, glacier, :V; aggregate = :mean,
        title = "V — $(label_for(key))")
    save(joinpath(kdir, "V_vs_obs.pdf"), fig_V)

    fig_Chist = plot_field_histogram(C;
        references = ["sliding = deformation (H=300m)" => C_scale,
            "0.5×crossover" => 0.5 * C_scale, "maxC" => maxC],
        mask = ice_mask_C(glacier), logScale = true,
        xlabel = "C", title = "Inverted C — $(label_for(key))")
    save(joinpath(kdir, "C_histogram.pdf"), fig_Chist)

    frac = Huginn.sliding_fraction(C, Huginn.inn1(glacier.H₀), A_value;
        p = law.p, q = law.q, basis = :surface)
    fig_frac = plot_field_histogram(frac;
        mask = ice_mask_C(glacier),
        xlabel = "sliding fraction (surface)", title = "Sliding share — $(label_for(key))")
    save(joinpath(kdir, "sliding_fraction_histogram.pdf"), fig_frac)
end

# Summary across configurations, Lisa's Fig. 6 idiom: one bar per configuration.
summary_stat = Dict(key => begin
    C = C_fields[key]
    A_value = is_ladder ? only(a for (n, a) in saved["ladder"] if n == key) : saved["A"]
    frac = Huginn.sliding_fraction(C, Huginn.inn1(glacier.H₀), A_value;
        p = law.p, q = law.q, basis = :surface)
    Statistics.median(filter(isfinite, frac[ice_mask_C(glacier)]))
end for key in keys_sorted)

fig_summary = Figure(size = (150 * length(keys_sorted) + 200, 450))
ax = Axis(fig_summary[1, 1], ylabel = "median sliding fraction (surface)",
    xticks = (1:length(keys_sorted), label_for.(keys_sorted)),
    title = "Sliding share across the run — $(RGI_ID), $(law.p == 3.0 && law.q == 0.0 ? "weertman" : "budd")")
barplot!(ax, 1:length(keys_sorted), [summary_stat[k] for k in keys_sorted])
save(joinpath(PLOTS_DIR, "summary_sliding_fraction.pdf"), fig_summary)

@info "Diagnostics written" dir=PLOTS_DIR
