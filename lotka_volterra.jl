using DifferentialEquations
using Plots
using Flux, DiffEqFlux

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

# Initial conditions, Time domain, Parameters
u₀ = [1.0,1.0]
timespan = (0.0,10.0)
parameters = [1.5,1.0,3.0,1.0]

# posing the ODE problem
problem = ODEProblem(lotka_volterra, u₀, timespan, parameters)

# solving the ODE problem
solution = solve(prob)
plot(solution)

# Now the inverse problem: try to find parameters so that the ODE solution is a cosine.
p = param([2.2, 1.0, 2.0, 0.4])
params = Flux.Params([p])

# one-layer neural network
function predict_rd()
  Tracker.collect(diffeq_rd(p, problem, Tsit5(), saveat=0.1))
end

# cost-function
loss_rd() = sum(abs2, x-cos(x) for x in predict_rd())

# optimalisation
data = Iterators.repeated((), 100)
opt = ADAM(0.1)
call_back = function () # callback function to observe training
  display(loss_rd())
  display(plot(solve(remake(problem, p = Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# training 
call_back()
Flux.train!(loss_rd, params, data, opt, cb = call_back)
@show p
