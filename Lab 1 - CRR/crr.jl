using LinearAlgebra
using Random
using GLM, StatsBase, DataFrames, Plots
plotlyjs()   # set the backend to Plotly

"""
Compute the prices at every node of the Bermudan option in the CRR model

- `prices` a squared  matrix with size `N+1` containing on output all the prices in the upper triangle
- `N` number of time steps
- `up` and `down` define the multiplicative moves in the tree. `p` is the probability of moving up
- `R` is the interest rate on a single time period
- `spot` is the initial value of the tree
- `x -> payoff(x)` returns the payoff value at `x`
"""
function crr!(prices::Matrix{<:Real}, N::Integer, up::Real, down::Real, p::Real, R::Real, spot::Real, payoff::Function)
    @assert size(prices) == (N+1, N+1) "prices must be of size N+1"
    # Initialize at maturity time
    X = spot * up^N
    discount = 1. / (1. + R)
    for i in 1:N+1
        prices[N+1, i] = payoff(X)
        X *= down / up
    end

    # Solve DDP
    for t in N-1:-1:0
        # The index inside the prices matrix
        timeIndex = t + 1
        X = spot * up^t
        for i in 1:t+1
            prices[timeIndex, i] = max(payoff(X), discount * (p * prices[timeIndex+1, i] + (1. - p) * prices[timeIndex+1, i+1]))
            X *= down / up
        end
    end
end

function crr(N::Integer, up::Real, down::Real, p::Real, R::Real, spot::Real, payoff::Function)
    prices = Matrix{Float64}(undef, N+1, N+1)
    crr!(prices, N, up, down, p, R, spot, payoff)
    return prices[1, 1]
end

struct CRRTree
    N::Integer
    up::Float64
    down::Float64
    spot::Float64
    prices::Matrix{Float64}
end

"""
Create a CRRTree using the discrete time parameters. When `isExtendedForDelta` is `true`,
the tree starts at time `-2` to enable the computation of the delta at each time step.
"""
function CRRTree(N::Integer, up::Float64, down::Float64, p::Float64, R::Float64, spot::Float64, payoff::Function)
    prices = Matrix{Float64}(undef, N + 1, N + 1)
    crr!(prices, N, up, down, p, R, spot, payoff)
    return CRRTree(N, up, down, spot, prices)
end

"""
Create a CRRTree using the continuous time (Black-Scholes) parameters. When `isExtendedForDelta` is `true`,
the tree starts at time `-2` to enable the computation of the delta at each time step.
"""
function CRRTree(T::Real, sigma::Real, r::Real, nSteps::Integer, spot::Real, payoff::Function)
    h = T / nSteps
    up = exp(sigma * sqrt(h))
    down = exp(-sigma * sqrt(h))
    R = exp(r * h) - 1.
    p = (down - exp(r * h)) / (down - up)
    return CRRTree(nSteps, up, down, p, R, spot, payoff)
end

"""
Compute the price on the CRR Tree at time `t` and position `s`
"""
function price(T::CRRTree, t::Integer, s::Real)
    @assert t >= 0 && t <= T.N "0 <= t <= N"
    X = T.spot * T.up^t
    sIndex = round(Int, log(s / X) / log(T.down / T.up))
    # Shift indices by 1 because matrix indexing starts at 1.
    return T.prices[t + 1, sIndex + 1]
end

"""
Compute the CRR approximation of the Black-Scholes American price

- `T` maturity
- `sigma` volatility
- `r` instantaneous interest rate
- `nSteps` number of time steps
- `spot` is the initial value of the tree
- `x -> payoff(x)` returns the payoff value at `x`
"""
function crrAmerPrice(T::Real, sigma::Real, r::Real, nSteps::Integer, spot::Real, payoff::Function)
    h = T / nSteps
    up = exp(sigma * sqrt(h))
    down = exp(-sigma * sqrt(h))
    R = exp(r * h) - 1.
    p = (down - exp(r * h)) / (down - up)
    return crr(nSteps, up, down, p, R, spot, payoff)
end





function asset!(path::Vector{<:Real}, N::Integer, up::Real, down::Real, p::Real, spot::Real, rng::AbstractRNG)
    @assert length(path) == N+1 "path must have length $N+1"
    path[1] = spot
    for i in 1:N
        move = rand(rng, Float64)
        path[i+1] = path[i] * (up * (move <= p) + down * (move > p))
    end
end

"""
Sample a path of the CRR tree in the Black-Scholes approximation

- `path` contains the sample path on exit. It must be of size `nSteps + 1` on input
- `T` maturity
- `sigma` volatility
- `r` instantaneous interest rate
- `nSteps` number of time steps
- `spot` is the initial value of the tree
- `x -> payoff(x)` returns the payoff value at `x`
"""
function asset!(path::Vector{<:Real}, T::Real, sigma::Real, r::Real, nSteps::Integer, spot::Real, rng::AbstractRNG)
    h = T / nSteps
    up = exp(sigma * sqrt(h))
    down = exp(-sigma * sqrt(h))
    p = (down - exp(r * h)) / (down - up)
    asset!(path, nSteps, up, down, p, spot, rng)
end

"""
Compute the terminal value of the delta portfolio
"""
function portfolio(crrTree::CRRTree, path::Vector{<:Real}, T::Real, r::Real, payoff::Function)
    dt = T / length(path)
    riskFreePortfolioComponent = price(crrTree, 0, path[1])
    prevDelta = 0.
    t = 0
    for st in path[1:end-1]
        # We exercise the option
        if abs(payoff(st) - price(crrTree, t, st)) < 1E-6
            return riskFreePortfolioComponent + prevDelta * st - payoff(st)
        end
        currentDelta = (price(crrTree, t + 1, st * crrTree.up) - price(crrTree, t + 1, st * crrTree.down)) / (st * (crrTree.up - crrTree.down))
        riskFreePortfolioComponent = riskFreePortfolioComponent * exp(r * dt) + (prevDelta - currentDelta) * st
        prevDelta = currentDelta
        t += 1
    end
    sT = path[end]
    return riskFreePortfolioComponent + prevDelta * sT - payoff(sT)
end

"""
Compute the hedging errors on `nSamples` paths
"""
function pnl(T::Real, sigma::Real, r::Real, nSteps::Integer, spot::Real, payoff::Function, nSamples::Integer, rng::AbstractRNG)
    errors = Vector{Float64}(undef, nSamples)
    crrTree = CRRTree(T, sigma, r, nSteps, spot, payoff)
    path = Vector{Float64}(undef, nSteps + 1)
    for i in 1:nSamples
        asset!(path, T, sigma, r, nSteps, spot, rng)
        errors[i] = portfolio(crrTree, path, T, r, payoff)
    end
    return errors
end


const S0 = 100.
const K = 100.
const r = 0.03
const sigma = 0.2
const T = 1.
putPayoff(x) = max(K - x, 0.)
callPayoff(x) = max(x - K, 0.)

function cvAmerPrice()
    refPrice = 6.742984
    # refPrice = crrAmerPrice(T, sigma, r, 300000, S0, putPayoff)

    N = [10, 100, 200, 500, 600, 800, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
    prices = crrAmerPrice.(T, sigma, r, N, S0, putPayoff)

    data = DataFrame(X=1.0 ./ N, Y=prices)
    crrCv = lm(@formula(Y ~ X), data)
    plot(N, GLM.coef(crrCv)[1] .+ GLM.coef(crrCv)[2] ./ N, label="regression", legend = :bottomright)
    display(scatter!(N, prices, label="prices"))
    GLM.r2(crrCv)

    println("Regression finds $(GLM.coef(crrCv)[1]) for the reference Price to be compared to $refPrice")
    println("Limt a = $(GLM.coef(crrCv)[2])")
end

function testPnl()
    errors = pnl(T, sigma, r, 10000, S0, putPayoff, 1000, MersenneTwister())
    println("Mean variance hedging error ", mean(errors .^ 2.0))
end

cvAmerPrice()
testPnl()
