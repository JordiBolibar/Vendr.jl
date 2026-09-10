"""
Environment setup for the gridded-C inversion campaign.

Points Vendr at the locally cloned ODINN-ecosystem packages. Huginn and Muninn are used
straight from `feature/mb-continuous-rhs`; ODINN and Sleipnir are used from the worktrees
that carry the two features this campaign needs, both branched off `feature/mb-continuous-rhs`:

  - ODINN    `feature/sliding-regularization`  spatial regularization of a gridded C
  - Sleipnir `feature/glathida-transient`      date aware glathida ingestion

Mass balance must be a continuous source term of the ice flow RHS for
`SciMLSensitivityAdjoint` to differentiate a transient run with MB, which is what
`feature/mb-continuous-rhs` provides.
"""

using Pkg

const DEPS = "/Users/Bolib001/Desktop/Jordi/Julia/Cinv-deps"

t0 = time()

Pkg.develop([
    PackageSpec(path = joinpath(DEPS, "worktrees", "sleipnir-glathida-transient")),
    PackageSpec(path = joinpath(DEPS, "Muninn")),
    PackageSpec(path = joinpath(DEPS, "Huginn")),
    PackageSpec(path = joinpath(DEPS, "worktrees", "odinn-sliding-reg")),
])

Pkg.instantiate()
Pkg.precompile()

@info "Environment resolved" elapsed_s = round(time() - t0; digits = 1)

Pkg.status()
