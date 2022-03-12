using JuMP, Ipopt, PyPlot

function mpcController(time, initialState, finalState, plotTraj=false)
    # Hyper parameters
    n = 30 # Time horizon
    dt = 0.1 # Timestep between constraints

    # Create the model
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)

    # Optimization variables
    @variable(model, state[1:n, 1:4])
    @variable(model, control[1:n, 1:2])

    # Initial conditions
    @constraint(model, state[1,1] == initialState[1])
    @constraint(model, state[1,2] == initialState[2])
    @constraint(model, state[1,3] == initialState[3])
    @constraint(model, state[1,4] == initialState[4])

    @constraint(model, control[:,2] .<= 100)
    @constraint(model, control[:,2] .>= -100)

    for t = 2:n
        @NLconstraint(model, state[t,1] == state[t-1,1] + dt*state[t-1,3]*cos(state[t-1,4]))
        @NLconstraint(model, state[t,2] == state[t-1,2] + dt*state[t-1,3]*sin(state[t-1,4]))
        @constraint(model, state[t,3] == state[t-1,3] + dt*control[t-1,1])
        @constraint(model, state[t,4] == state[t-1,4] + dt*control[t-1,2])
    end

    @constraint(model, state[n,:] .== finalState)

    controlWeight = 1.0
    @objective(model, Min, controlWeight*sum(control[:,1].^2))

    optimize!(model)

    if(plotTraj)
        states = value.(state)
        figure(figsize=(6,5))
        plot(states[:,1], states[:,2])
        title("Trajectory")
        axis([-5, 5, -5, 5])
    end

    return value.(control)[1,:]
end
