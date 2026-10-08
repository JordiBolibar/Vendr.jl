"""
Switch Sleipnir's `ODINN_prepro` artifact between the pinned published tarball (v0.0.3,
HuggingFace) and the live local Gungnir cache (`~/.ODINN/ODINN_prepro`).

Sleipnir resolves glacier data through `artifact"ODINN_prepro"`, which Julia's package
manager pins to one frozen, content-addressed tarball -- local Gungnir preprocessing (new
glaciers, regenerated bands like Millan22's per-pixel error) never reaches it without this
override, and it never will without a full publish-a-new-tarball cycle. This script edits
the machine-wide `~/.julia/artifacts/Overrides.toml` (not scoped to this repo, or even to
Julia -- it affects every project on this machine using the same Sleipnir artifact) to point
the artifact's tree-hash at the local directory instead.

Override key format matters: a hash override (this one) needs a *direct string* value —
`"<hash>" = "<path>"` — not the nested-table form (`[<hash>]` with `name = path` inside),
which Julia's Artifacts.jl parses as a package-UUID override instead and silently rejects
here since our key isn't a valid UUID. Uses TOML.jl and only ever touches this one key, so
any other override already in that file is left alone.

Run with:
    julia +1.11 --project=. toggle_local_prepro.jl on   # use ~/.ODINN/ODINN_prepro (live);
    #   overrides to ~/.ODINN (one level up), since prepro_dir() joins "ODINN_prepro" itself
    julia +1.11 --project=. toggle_local_prepro.jl off  # use the pinned v0.0.3 tarball
    julia +1.11 --project=. toggle_local_prepro.jl status
"""

using TOML

# The tree-hash Sleipnir's Artifacts.toml pins ODINN_prepro to. If Sleipnir bumps to a new
# published version, this hash (and the `git-tree-sha1` it's copied from) both need updating.
const TREE_HASH = "a67e063a09b81a66b21dd2e74f780dc98e6d4915"
const LOCAL_PREPRO = joinpath(homedir(), ".ODINN")
const LOCAL_PREPRO_DATA = joinpath(LOCAL_PREPRO, "ODINN_prepro")
const OVERRIDES_PATH = joinpath(homedir(), ".julia", "artifacts", "Overrides.toml")

function load_overrides()
    isfile(OVERRIDES_PATH) ? TOML.parsefile(OVERRIDES_PATH) : Dict{String, Any}()
end

function save_overrides(d::Dict)
    mkpath(dirname(OVERRIDES_PATH))
    open(OVERRIDES_PATH, "w") do io
        TOML.print(io, d)
    end
end

mode = isempty(ARGS) ? "status" : ARGS[1]

overrides = load_overrides()
active = get(overrides, TREE_HASH, nothing) == LOCAL_PREPRO

if mode == "status"
    println(active ? "ON  (using $(LOCAL_PREPRO_DATA))" : "OFF (using the pinned v0.0.3 tarball)")
elseif mode == "on"
    isdir(LOCAL_PREPRO_DATA) || error("$(LOCAL_PREPRO_DATA) does not exist -- run Gungnir preprocessing first.")
    overrides[TREE_HASH] = LOCAL_PREPRO
    save_overrides(overrides)
    println("ON: artifact\"ODINN_prepro\" now resolves to $(LOCAL_PREPRO_DATA)")
elseif mode == "off"
    delete!(overrides, TREE_HASH)
    save_overrides(overrides)
    println("OFF: artifact\"ODINN_prepro\" resolves to the pinned v0.0.3 tarball again")
else
    error("usage: toggle_local_prepro.jl [on|off|status]")
end
