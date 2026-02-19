module MacroPlannerOracle

# Oracle planner: Perfect information baseline (no POMDPs needed!)
# Has access to ground truth and other agents' plans
using Random
using LinearAlgebra
using ..Types
import ..Types: check_battery_feasible, simulate_battery_evolution
# Import types from the parent module (Planners)
import ..EventState, ..NO_EVENT, ..EVENT_PRESENT
import ..Agent, ..SensingAction, ..GridObservation, ..EventMap
# Import trajectory functions
import ..Agents.TrajectoryPlanner.get_position_at_time

export best_script, evaluate_action_sequence_exact, calculate_macro_script_reward

"""
best_script(env, belief, agent::Agent, C::Int, other_scripts, gs_state)::Vector{SensingAction}
  – Oracle planner with perfect information (online/reactive)
  – Returns a reactive policy (not a script!) that checks ground truth at execution time
  – Decision rule at each step:
    1. Look at CURRENT ground truth (not predicted!)
    2. If event in FOR → observe it (prioritize: least observed → longest since obs → random)
  – Note: This is online decision-making, not offline planning
"""
function best_script(env, belief, agent, C::Int, other_scripts, gs_state; rng::AbstractRNG=Random.GLOBAL_RNG)
    # Start timing
    start_time = time()
    
    println("🔮 Oracle planner: Enabling online decision-making with perfect information")
    
    # Store environment reference in agent so trajectory_planner can access it
    agent.env_ref = env
    
    # Oracle doesn't create a plan or policy - it makes decisions online in trajectory_planner.jl
    # The ground station will set agent_plan_types[agent_id] = :oracle
    # Then execute_plan in trajectory_planner will handle the online logic
    
    # End timing
    end_time = time()
    planning_time = end_time - start_time
    
    println("✅ Oracle enabled for Agent $(agent.id) in $(round(planning_time, digits=3)) seconds")
    
    # Return empty sequence - oracle makes decisions at execution time
    return SensingAction[], planning_time
end

"""
Select best cell to observe based on priority rule:
1. Least observed (lowest observation count)
2. If tie: longest time since last observation
3. If still tie: random
"""
function select_best_cell_to_observe(event_cells::Vector{Tuple{Int, Int}}, 
                                     observation_history::Dict{Tuple{Int, Int}, Tuple{Int, Int}},
                                     current_time::Int,
                                     rng::AbstractRNG)
    if length(event_cells) == 1
        return event_cells[1]
    end
    
    # Calculate scores for each cell
    cell_scores = []
    for cell in event_cells
        if haskey(observation_history, cell)
            obs_count, last_obs_time = observation_history[cell]
            time_since_obs = current_time - last_obs_time
        else
            # Never observed before - highest priority
            obs_count = 0
            time_since_obs = typemax(Int)  # Infinite time since last obs
        end
        push!(cell_scores, (cell, obs_count, time_since_obs))
    end
    
    # Sort by: 1) least observed, 2) longest since last obs
    sort!(cell_scores, by = x -> (x[2], -x[3]))
    
    # Check for ties at the best level
    best_obs_count = cell_scores[1][2]
    best_time_since = cell_scores[1][3]
    
    # Find all cells with same best score
    tied_cells = [x[1] for x in cell_scores if x[2] == best_obs_count && x[3] == best_time_since]
    
    if length(tied_cells) == 1
        return tied_cells[1]
    else
        # Break tie randomly
        return rand(rng, tied_cells)
    end
end

"""
Get ground truth state from environment
Returns a matrix indicating the true event state at each cell
"""
function get_ground_truth_state(env)
    height, width = env.height, env.width
    ground_truth = Matrix{Int}(undef, height, width)
    
    # Try multiple possible locations for ground truth
    ground_truth_matrix = nothing
    
    # Option 1: env.current_state (if your main script sets this)
    if hasproperty(env, :current_state) && env.current_state !== nothing
        ground_truth_matrix = env.current_state
        
    # Option 2: env.event_map (Matrix{EventState} directly)  
    elseif hasproperty(env, :event_map) && env.event_map !== nothing
        ground_truth_matrix = env.event_map
        
    # Option 3: Try to get from GridState if env has a state field
    elseif hasproperty(env, :state) && hasproperty(env.state, :event_map)
        ground_truth_matrix = env.state.event_map
        
    else
        # ERROR: No ground truth available!
        error("❌ Oracle Error: No ground truth found! The environment must have one of: env.current_state, env.event_map, or env.state.event_map")
    end
    
    # Convert EventState matrix to integer matrix
    for y in 1:height, x in 1:width
        cell_state = ground_truth_matrix[y, x]
        
        # Convert to integer: NO_EVENT = 1, EVENT_PRESENT = 2
        if cell_state == NO_EVENT
            ground_truth[y, x] = 1
        elseif cell_state == EVENT_PRESENT
            ground_truth[y, x] = 2
        else
            # Default to no event if unknown
            ground_truth[y, x] = 1
        end
    end
    
    return ground_truth
end

"""
Evaluate a sequence using perfect information (oracle knowledge)
"""
function evaluate_sequence_with_oracle(sequence::Vector{SensingAction}, agent, env, gs_state, 
                                       ground_truth, other_scripts, C::Int)
    γ = env.discount
    total_reward = 0.0
    
    # Current time for this agent
    current_time = gs_state.time_step
    
    # Simulate forward with perfect information
    simulated_ground_truth = copy(ground_truth)
    
    for (k, action) in enumerate(sequence)
        t_global = current_time + k - 1
        
        # Calculate reward for this action with perfect information
        step_reward = calculate_oracle_reward(action, agent, simulated_ground_truth, 
                                             t_global, env, gs_state, other_scripts)
        
        # Apply discount
        total_reward += (γ^(k-1)) * step_reward
        
        # Simulate ground truth evolution (events can spread)
        simulated_ground_truth = simulate_ground_truth_evolution(simulated_ground_truth, env)
    end
    
    return total_reward
end

"""
Calculate reward with oracle knowledge
The oracle knows:
1. True state of each cell
2. What other agents will observe
3. Future ground truth evolution
"""
function calculate_oracle_reward(action::SensingAction, agent, ground_truth, 
                                t_global::Int, env, gs_state, other_scripts)
    reward = 0.0
    
    if isempty(action.target_cells)
        # Wait action - no reward
        return 0.0
    end
    
    # For each target cell, calculate value based on ground truth
    for cell in action.target_cells
        x, y = cell
        true_state = ground_truth[y, x]
        
        # Reward strategy: prioritize cells with events (true state = EVENT_PRESENT)
        # Also consider if other agents will observe this cell
        cell_reward = 0.0
        
        if true_state == 2  # EVENT_PRESENT
            # High reward for detecting actual events
            cell_reward += 1.0
            
            # Check if any other agent will observe this cell at this time
            # If not, give extra reward for being the only one to observe
            if !will_other_agent_observe(cell, t_global, other_scripts, gs_state, env)
                cell_reward += 0.5  # Bonus for unique coverage
            end
        else
            # Lower reward for observing no-event cells
            # But still valuable for reducing uncertainty
            cell_reward += 0.1
            
            # Penalize redundant observations
            if will_other_agent_observe(cell, t_global, other_scripts, gs_state, env)
                cell_reward -= 0.05  # Small penalty for redundancy
            end
        end
        
        reward += cell_reward
    end
    
    return reward
end

"""
Check if any other agent will observe a cell at a given time
"""
function will_other_agent_observe(cell::Tuple{Int, Int}, t_global::Int, 
                                 other_scripts, gs_state, env)
    # Check all other agents' plans
    for (agent_id, plan) in gs_state.agent_plans
        if plan !== nothing
            # Calculate which action this agent will take at t_global
            plan_timestep = (t_global - gs_state.agent_last_sync[agent_id]) + 1
            
            if 1 <= plan_timestep <= length(plan)
                action = plan[plan_timestep]
                if cell in action.target_cells
                    return true
                end
            end
        end
    end
    
    return false
end

"""
Simulate ground truth evolution based on environment dynamics
"""
function simulate_ground_truth_evolution(ground_truth, env)
    # Create evolved ground truth
    evolved = copy(ground_truth)
    height, width = size(ground_truth)
    
    # Apply RSP dynamics: events can spread to neighbors
    for y in 1:height, x in 1:width
        if ground_truth[y, x] == 2  # EVENT_PRESENT
            # Event can spread to neighbors based on RSP parameters
            cell_params = Types.get_cell_rsp_params(env.rsp_params, y, x)
            
            # Check all 8 neighbors
            for dx in -1:1, dy in -1:1
                if dx == 0 && dy == 0
                    continue
                end
                
                nx, ny = x + dx, y + dy
                if 1 <= nx <= width && 1 <= ny <= height
                    # Probability of contagion
                    if rand() < cell_params.alpha
                        evolved[ny, nx] = 2  # Spread event
                    end
                end
            end
        else
            # No event: check for spontaneous ignition
            cell_params = Types.get_cell_rsp_params(env.rsp_params, y, x)
            if rand() < (cell_params.beta0 + cell_params.lambda)
                evolved[y, x] = 2  # Spontaneous ignition
            end
        end
    end
    
    return evolved
end

"""
Generate all possible action sequences of length C
"""
function generate_action_sequences(agent, env, C::Int, gs_state)
    if C == 0
        return Vector{SensingAction}[]
    end
    
    # Generate actions for each timestep
    actions_per_timestep = Vector{Vector{SensingAction}}()
    
    for t in 1:C
        global_time = gs_state.time_step + t - 1
        pos = get_position_at_time(agent.trajectory, global_time, agent.phase_offset)
        for_cells = get_field_of_regard_at_position(agent, pos, env)
        
        # Generate actions for this timestep
        timestep_actions = SensingAction[]
        
        # Add wait action
        push!(timestep_actions, SensingAction(agent.id, Tuple{Int, Int}[], false))
        
        # Add single-cell sensing actions
        for cell in for_cells
            action = SensingAction(agent.id, [cell], false)
            if check_battery_feasible(agent, action, agent.battery_level)
                push!(timestep_actions, action)
            end
        end
        
        push!(actions_per_timestep, timestep_actions)
    end
    
    # Generate all combinations
    sequences = generate_sequences_from_actions_per_timestep(actions_per_timestep)
    
    return sequences
end

"""
Generate all sequences by selecting one action per timestep
"""
function generate_sequences_from_actions_per_timestep(actions_per_timestep::Vector{Vector{SensingAction}})
    if isempty(actions_per_timestep)
        return Vector{SensingAction}[]
    elseif length(actions_per_timestep) == 1
        return [[action] for action in actions_per_timestep[1]]
    else
        sequences = Vector{SensingAction}[]
        
        # Get actions for current timestep
        current_actions = actions_per_timestep[1]
        
        # Recursively generate sequences for remaining timesteps
        remaining_sequences = generate_sequences_from_actions_per_timestep(actions_per_timestep[2:end])
        
        # Combine current actions with remaining sequences
        for action in current_actions
            for remaining_seq in remaining_sequences
                new_seq = [action; remaining_seq]
                push!(sequences, new_seq)
            end
        end
        
        return sequences
    end
end

"""
Evaluate action sequence (for compatibility with other planners)
Note: belief parameter ignored - oracle uses ground truth
"""
function evaluate_action_sequence_exact(env, belief₀, agent, seq, other_scripts, C, gs_state, rng::AbstractRNG)
    ground_truth = get_ground_truth_state(env)
    return evaluate_sequence_with_oracle(seq, agent, env, gs_state, ground_truth, other_scripts, C)
end

"""
Calculate reward for macro-script (for compatibility with other planners)
"""
function calculate_macro_script_reward(seq::Vector{SensingAction}, other_scripts, C::Int, env, agent, B_branches, gs_state)
    ground_truth = get_ground_truth_state(env)
    return evaluate_sequence_with_oracle(seq, agent, env, gs_state, ground_truth, other_scripts, C)
end

"""
Get field of regard for an agent at a specific position
"""
function get_field_of_regard_at_position(agent, position, env)
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    
    # Check sensor pattern
    if agent.sensor.pattern == :cross
        # Cross-shaped sensor: agent's position and adjacent cells
        ax, ay = position
        for dx in -1:1, dy in -1:1
            nx, ny = ax + dx, ay + dy
            if 1 <= nx <= env.width && 1 <= ny <= env.height
                # Only include cross pattern (not diagonal)
                if (dx == 0 && dy == 0) || (dx == 0 && dy != 0) || (dx != 0 && dy == 0)
                    push!(fov_cells, (nx, ny))
                end
            end
        end
    elseif agent.sensor.pattern == :circular
        # Circular sensor: agent's position and all 8 adjacent cells
        for dx in -1:1, dy in -1:1
            nx, ny = x + dx, y + dy
            if 1 <= nx <= env.width && 1 <= ny <= env.height
                push!(fov_cells, (nx, ny))
            end
        end
    elseif agent.sensor.pattern == :row_only || agent.sensor.range == 0.0
        # Row-only visibility: agent can only see cells in its current row
        for nx in 1:env.width
            push!(fov_cells, (nx, y))
        end
    else
        # Standard sensor range visibility
        sensor_range = round(Int, agent.sensor.range)
        for dx in -sensor_range:sensor_range
            for dy in -sensor_range:sensor_range
                nx, ny = x + dx, y + dy
                if 1 <= nx <= env.width && 1 <= ny <= env.height
                    # Check if within sensor range
                    distance = sqrt(dx^2 + dy^2)
                    if distance <= agent.sensor.range
                        push!(fov_cells, (nx, ny))
                    end
                end
            end
        end
    end
    return fov_cells
end

end # module

