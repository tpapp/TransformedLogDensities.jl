"""
Placeholder for a short summary about TransformedLogDensities.
"""
module TransformedLogDensities

export TransformedLogDensity

import LogDensityProblems
import TransformVariables
using UnPack: @unpack

"""
    TransformedLogDensity(transformation, log_density_function)

A problem in Bayesian inference. Vectors of length compatible with the dimension (obtained
from `transformation`) are transformed into a general object `θ` (unrestricted type, but a
named tuple is recommended for clean code), correcting for the log Jacobian determinant of
the transformation.

`log_density_function(θ)` is expected to return *real numbers*. For zero densities or
infeasible `θ`s, `-Inf` or similar should be returned, but for efficiency of inference most
methods recommend using `transformation` to avoid this. It is recommended that
`log_density_function` is a callable object that also encapsulates the data for the problem.

Use the property accessors `ℓ.transformation` and `ℓ.log_density_function` to access the
arguments of `ℓ::TransformedLogDensity`, these are part of the public API.

# Usage note

This is the most convenient way to define log densities, as `capabilities`, `logdensity`,
and `dimension` are automatically defined. To obtain a type that supports derivatives, use
[`ADgradient`](@ref).
"""
struct TransformedLogDensity{T <: TransformVariables.AbstractTransform, L}
    transformation::T
    log_density_function::L
end

function Base.show(io::IO, ℓ::TransformedLogDensity)
    d = LogDensityProblems.dimension(ℓ)
    print(io, "TransformedLogDensity of dimension $(d)")
end

function LogDensityProblems.capabilities(::Type{<:TransformedLogDensity})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(p::TransformedLogDensity)
    TransformVariables.dimension(p.transformation)
end

function LogDensityProblems.logdensity(p::TransformedLogDensity, x::AbstractVector)
    @unpack transformation, log_density_function = p
    TransformVariables.transform_logdensity(transformation, log_density_function, x)
end

end # module
