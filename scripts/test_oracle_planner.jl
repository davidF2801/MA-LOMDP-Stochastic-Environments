"""
Test script for Oracle Planner
Demonstrates how to use the oracle planner as an upper bound baseline
"""

# Add the project to the load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using MyProject
using MyProject.Types
using MyProject.Environment
using MyProject.Agents
using MyProject.Planners
using Random

println("=" ^ 80)
println("ORACLE PLANNER TEST")
println("=" ^ 80)

# Set random seed for reproducibility
Random.seed!(42)

# Create a simple test environment
println("\n📋 Setting up test environment...")

# Environment parameters
grid_size = (7, 7)  # 7x7 grid
num_agents = 2

# RSP Parameters for event dynamics
rsp_params = Types.RSPParameters(
    alpha = 0.3,    # Contagion rate
    beta0 = 0.01,   # Spontaneous ignition
    gamma = 0.1,    # Recovery rate
    lambda = 0.05   # External ignition
)

# Create environment
env = Environment.create_environment(
    width = grid_size[1],
    height = grid_size[2],
    rsp_params = rsp_params,
    discount = 0.95,
    max_sensing_targets = 1,
    dynamics = :rsp
)

# Create agents with linear trajectories
println("👥 Creating agents...")
agents = Dict{Int, Agent}()

for i in 1:num_agents
    # Agent moves along row i
    trajectory = LinearTrajectory(
        start_pos = (1, i),
        end_pos = (grid_size[1], i),
        period = grid_size[1]
    )
    
    sensor = RangeLimitedSensor(
        range = 0.0,  # Row-only sensor
        pattern = :row_only
    )
    
    agent = Agent(
        id = i,
        trajectory = trajectory,
        sensor = sensor,
        battery_capacity = 100.0,
        battery_level = 100.0,
        phase_offset = (i-1) * 2  # Stagger agents
    )
    
    agents[i] = agent
end

# Initialize ground truth with some events
println("🔥 Initializing ground truth with events...")
for y in 1:grid_size[2], x in 1:grid_size[1]
    if rand() < 0.2  # 20% chance of event
        env.event_map.states[y, x] = EVENT_PRESENT
        println("  Event at ($x, $y)")
    end
end

# Initialize ground station state
println("\n🛰️ Initializing ground station...")
gs_state = Planners.GroundStation.GroundStationState(
    time_step = 0,
    agent_last_sync = Dict(i => -1 for i in 1:num_agents),
    agent_plans = Dict{Int, Union{Nothing, Vector{SensingAction}}}(i => nothing for i in 1:num_agents),
    agent_observation_history = Dict{Int, Vector{Tuple{Int, Tuple{Int,Int}, EventState}}}(i => [] for i in 1:num_agents),
    global_belief = Planners.GroundStation.initialize_global_belief(env)
)

# Planning horizon
C = 5  # Plan 5 steps ahead

println("\n" * "=" ^ 80)
println("COMPARING PLANNERS")
println("=" ^ 80)

# Test different planning modes
planning_modes = [:oracle, :sweep, :random]

results = Dict{Symbol, Any}()

for mode in planning_modes
    println("\n" * "-" ^ 80)
    println("Testing: $mode")
    println("-" ^ 80)
    
    agent = agents[1]
    agent_belief = gs_state.global_belief
    other_scripts = Dict{Int, Vector{SensingAction}}()
    
    # Select planner based on mode
    if mode == :oracle
        planner = Planners.MacroPlannerOracle
        println("🔮 Using Oracle Planner (perfect information)")
    elseif mode == :sweep
        planner = Planners.MacroPlannerSweep
        println("🔄 Using Sweep Planner (baseline)")
    elseif mode == :random
        planner = Planners.MacroPlannerRandom
        println("🎲 Using Random Planner (baseline)")
    else
        println("❌ Unknown planner mode: $mode")
        continue
    end
    
    # Plan with this planner
    try
        best_sequence, planning_time = planner.best_script(
            env,
            agent_belief,
            agent,
            C,
            other_scripts,
            gs_state
        )
        
        println("\n✅ Planning complete!")
        println("  Planning time: $(round(planning_time, digits=3))s")
        println("  Sequence length: $(length(best_sequence))")
        println("  Actions:")
        
        for (t, action) in enumerate(best_sequence)
            if isempty(action.target_cells)
                println("    Step $t: WAIT")
            else
                cells_str = join(["($x,$y)" for (x,y) in action.target_cells], ", ")
                println("    Step $t: SENSE at $cells_str")
            end
        end
        
        # Evaluate the sequence
        if mode == :oracle
            ground_truth = Planners.MacroPlannerOracle.get_ground_truth_state(env)
            value = Planners.MacroPlannerOracle.evaluate_sequence_with_oracle(
                best_sequence, agent, env, gs_state, ground_truth, other_scripts, C
            )
        else
            # For other planners, use a simple evaluation
            value = length([a for a in best_sequence if !isempty(a.target_cells)]) * 0.5
        end
        
        results[mode] = Dict(
            :sequence => best_sequence,
            :planning_time => planning_time,
            :value => value
        )
        
        println("  Estimated value: $(round(value, digits=3))")
        
    catch e
        println("❌ Error with $mode planner: $e")
        println(stacktrace(catch_backtrace()))
    end
end

# Compare results
println("\n" * "=" ^ 80)
println("PERFORMANCE COMPARISON")
println("=" ^ 80)

if haskey(results, :oracle)
    oracle_value = results[:oracle][:value]
    println("\n📊 Oracle (Upper Bound): $(round(oracle_value, digits=3))")
    
    for mode in [:sweep, :random]
        if haskey(results, mode)
            planner_value = results[mode][:value]
            efficiency = planner_value / oracle_value * 100
            gap = oracle_value - planner_value
            
            println("\n$mode:")
            println("  Value: $(round(planner_value, digits=3))")
            println("  Efficiency: $(round(efficiency, digits=1))% of oracle")
            println("  Gap from optimal: $(round(gap, digits=3))")
        end
    end
else
    println("\n❌ Oracle results not available for comparison")
end

println("\n" * "=" ^ 80)
println("INTERPRETATION")
println("=" ^ 80)
println("""
The Oracle planner represents the THEORETICAL UPPER BOUND - what's achievable
with perfect information about:
  • Ground truth state of all cells
  • Other agents' planned actions
  • Future event evolution

The performance gap between Oracle and other planners shows how much reward
is lost due to uncertainty and imperfect coordination.

Key Insights:
  • Oracle should have the highest value (it's optimal with perfect info)
  • Smaller gaps indicate better handling of uncertainty
  • Large gaps suggest room for improvement in sensing strategy

Use Oracle as a benchmark to evaluate how close your planners get to optimal!
""")

println("\n✅ Test complete!")


