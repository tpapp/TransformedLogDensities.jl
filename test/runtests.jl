using TransformedLogDensities, TransformVariables, Test, Distributions
using LogDensityProblems: LogDensityProblems, capabilities, logdensity,
    logdensity_and_gradient, LogDensityOrder
using LogDensityProblemsAD: ADgradient
import ForwardDiff
using LogDensityProblems: random_reals # needs to be imported explicitly

@testset "-∞ log densities" begin
    t = as(Array, 2)
    validx = x -> all(x .> 0)
    p = TransformedLogDensity(t, x -> validx(x) ?  sum(abs2, x)/2 : -Inf)
    ∇p = ADgradient(:ForwardDiff, p)

    @test LogDensityProblems.dimension(p) == LogDensityProblems.dimension(∇p) == TransformVariables.dimension(t)
    @test p.transformation ≡ parent(∇p).transformation ≡ t

    for _ in 1:100
        x = random_reals(dimension(t))
        px = logdensity(∇p, x)
        px_∇px = logdensity_and_gradient(∇p, x)
        @test px isa Real
        @test px_∇px isa Tuple{Real,Any}
        @test first(px_∇px) ≡ px
        if validx(x)
            @test px ≈ sum(abs2, x)/2
            @test last(px_∇px) ≈ x
        else
            @test isinf(px)
        end
    end
end

@testset "transformed Bayesian problem" begin
    t = as((y = asℝ₊, ))
    d = LogNormal(1.0, 2.0)
    logposterior = ((x, ), ) -> logpdf(d, x)

    # a Bayesian problem
    p = TransformedLogDensity(t, logposterior)
    @test repr(p) == "TransformedLogDensity of dimension 1"
    @test LogDensityProblems.dimension(p) == 1
    @test p.transformation ≡ t
    @test capabilities(p) == LogDensityOrder(0)

    # gradient of a problem
    ∇p = ADgradient(:ForwardDiff, p)
    @test LogDensityProblems.dimension(∇p) == 1
    @test parent(∇p).transformation ≡ t

    for _ in 1:100
        x = random_reals(dimension(t))
        θ, lj = transform_and_logjac(t, x)
        px = logdensity(p, x)
        @test logpdf(d, θ.y) + lj ≈ (px::Real)
        px2, ∇px = logdensity_and_gradient(∇p, x)
        @test px2 == px
        @test ∇px ≈ [ForwardDiff.derivative(x -> logpdf(d, exp(x)) + x, x[1])]
    end
end
