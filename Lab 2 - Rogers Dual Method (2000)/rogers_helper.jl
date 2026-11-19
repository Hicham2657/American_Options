using Printf
using Random: randn!
using Statistics, Distributions
using Optim

"""
Compute one simulation of the BS model with n time steps over [0,T]

# Arguments
- `path` an array of size J + 1, containing a simulation of the model on output
- `G`: an array of size J filled with normal random variables.
"""
function bsAsset!(path::Array{Float64}, sigma::Float64, r::Float64, T::Float64, spot::Float64, n::Int, G::Array{Float64})
    @assert length(path) == n + 1
    dt = T / n
    sqrt_dt = sqrt(dt)
    path[1] = spot
    for i in 1:n
        path[i + 1] = path[i] * exp((r - sigma * sigma / 2) * dt + sigma * (sqrt_dt * G[i]))
    end
end

"""
Compute the price at time 0 of European put option
"""
function pricePutEuro(spot::Float64, sigma::Float64, r::Float64, T::Float64, strike::Float64)
    d1 = (log(spot / strike) + (r + sigma * sigma / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return - spot * cdf(Normal(), -d1) + strike * exp(-r * T) * cdf(Normal(), -d2)
end

"""
Compute the delta at time 0 of European put option
"""
function deltaPutEuro(spot::Float64, sigma::Float64, r::Float64, T::Float64, strike::Float64)
    d1 = (log(spot / strike) + (r + sigma * sigma / 2) * T) / (sigma * sqrt(T))
    return - cdf(Normal(), -d1)
end

"""
Compute the payoff at each date on an asset path
- `Z` contains the payoff path on output. Must be the same length as `path`
- `path` a model path.
"""
function putPayoff!(Z::Array{Float64}, path::Array{Float64}, strike::Float64)
    @assert length(Z) == length(path) "Z and path must have the same size"
    Z .= max.(strike .- path, 0.)
end

"""
Generate `nSamples` of M, Z, S where
    - S is the underlying asset
    - M is the martingale computed on the path of `S`
    - Z is the discounted payoff computed on the path of `S`

Return (M, Z, S)
"""
function generateSamples(spot::Float64, sigma::Float64, r::Float64, T::Float64, strike::Float64, nDates::Int, nSamples::Int, martingale!::Function)
    path = Vector{Vector{Float64}}(undef, nSamples)
    M = Vector{Vector{Float64}}(undef, nSamples)
    Z = Vector{Vector{Float64}}(undef, nSamples)
    G = Vector{Float64}(undef, nDates)
    dates = collect(T / nDates * range(0, nDates))
    for i in 1:nSamples
        # To be completed
    end
    return (M, Z, path)
end