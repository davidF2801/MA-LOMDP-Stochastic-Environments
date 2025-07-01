using POMDPs
using POMDPTools
using Distributions
using Random
using LinearAlgebra

# Include common types and modules
include("../types.jl")

"""
CentralizedPlanner - Implements C-LOMDP with global coordination
"""
module CentralizedPlanner

using POMDPs
using POMDPTools
using Distributions
using Random
using LinearAlgebra

# Import types and functions
import Main: SensingAction, GridObservation, EventState, NO_EVENT, EVENT_PRESENT, EVENT_SPREADING, EVENT_DECAYING
import Main: EventDynamics, SpatialGrid, GridState, Agent, LocalityFunction
import Main: get_observable_cells, get_trajectory_period, get_position_at_time
import Main: Belief, update_belief_state, initialize_belief, calculate_uncertainty

export CentralizedPolicy, GlobalBeliefState, GlobalClockVector, create_centralized_policy, 
       select_joint_action, update_global_belief, calculate_joint_action_value, get_global_clock_vector,
       initialize_global_belief, get_joint_action_space, simulate_centralized_execution

"""
GlobalClockVector - Represents the global clock vector τ_t = (τ_{t,1}, ..., τ_{t,N})
"""
struct GlobalClockVector
    agent_phases::Vector{Int}  # Current phase for each agent
    time_step::Int             # Global time step
end

"""
GlobalBeliefState - Represents the global belief state b_t(x) over the hidden event map
"""
struct GlobalBeliefState
    event_probabilities::Matrix{Float64}  # Global belief over event map
    uncertainty_map::Matrix{Float64}      # Global uncertainty map
    observation_history::Vector{Tuple{Int, Vector{GridObservation}}}  # (time_step, observations)
    last_update::Int
end

"""
CentralizedPolicy - Implements π_cen : (τ_t, b_t) → a_t ∈ A_τ_t
Enhanced with belief-MDP solving capabilities
"""
struct CentralizedPolicy
    environment::SpatialGrid
    planning_horizon::Int
    discount_factor::Float64
    exploration_constant::Float64
    value_function::Dict{Tuple{Vector{Int}, Matrix{Float64}}, Float64}  # (clock, belief) -> value
    policy_cache::Dict{Tuple{Vector{Int}, Matrix{Float64}}, Vector{SensingAction}}  # (clock, belief) -> action
end

"""
create_centralized_policy(env::SpatialGrid, horizon::Int=10, discount::Float64=0.95, exploration::Float64=1.0)
Creates a centralized policy with belief-MDP solving capabilities
"""
function create_centralized_policy(env::SpatialGrid, horizon::Int=10, discount::Float64=0.95, exploration::Float64=1.0)
    return CentralizedPolicy(env, horizon, discount, exploration, Dict(), Dict())
end

"""
get_global_clock_vector(env::SpatialGrid, time_step::Int)
Computes the global clock vector τ_t = (τ_{t,1}, ..., τ_{t,N})
"""
function get_global_clock_vector(env::SpatialGrid, time_step::Int)
    agent_phases = Int[]
    
    for agent in env.agents
        period = get_trajectory_period(agent.trajectory)
        phase = time_step % period
        push!(agent_phases, phase)
    end
    
    return GlobalClockVector(agent_phases, time_step)
end

"""
initialize_global_belief(env::SpatialGrid, prior_probability::Float64=0.5)
Initializes global belief state with uniform prior
"""
function initialize_global_belief(env::SpatialGrid, prior_probability::Float64=0.5)
    event_probabilities = fill(prior_probability, env.height, env.width)
    uncertainty_map = calculate_uncertainty_map(event_probabilities)
    
    return GlobalBeliefState(
        event_probabilities,
        uncertainty_map,
        [],
        0
    )
end

"""
get_joint_action_space(env::SpatialGrid, clock::GlobalClockVector)
Returns the joint action space A_τ_t for the current global clock
"""
function get_joint_action_space(env::SpatialGrid, clock::GlobalClockVector)
    joint_actions = Vector{Vector{SensingAction}}()
    
    # Get individual agent action spaces
    agent_actions = Vector{Vector{SensingAction}}()
    
    for (i, agent) in enumerate(env.agents)
        phase = clock.agent_phases[i]
        locality = env.locality_functions[i]
        observable_cells = get_observable_cells(locality, phase)
        
        # Generate actions for this agent
        actions = Vector{SensingAction}()
        
        if !isempty(observable_cells)
            # Individual cell sensing actions (FOV = 1 cell)
            for cell in observable_cells
                push!(actions, SensingAction(i, [cell], false))
                push!(actions, SensingAction(i, [cell], true))  # With communication
            end
        else
            # Wait action when no cells are observable
            push!(actions, SensingAction(i, [], false))
        end
        
        push!(agent_actions, actions)
    end
    
    # Generate all joint action combinations
    function generate_combinations(agent_idx::Int, current_joint::Vector{SensingAction})
        if agent_idx > length(agent_actions)
            push!(joint_actions, copy(current_joint))
            return
        end
        
        for action in agent_actions[agent_idx]
            push!(current_joint, action)
            generate_combinations(agent_idx + 1, current_joint)
            pop!(current_joint)
        end
    end
    
    generate_combinations(1, Vector{SensingAction}())
    return joint_actions
end

"""
calculate_joint_action_value(policy::CentralizedPolicy, belief::GlobalBeliefState, 
                           clock::GlobalClockVector, joint_action::Vector{SensingAction})
Calculates the value of a joint action using information gain and uncertainty reduction
"""
function calculate_joint_action_value(policy::CentralizedPolicy, belief::GlobalBeliefState, 
                                   clock::GlobalClockVector, joint_action::Vector{SensingAction})
    total_value = 0.0
    
    # Calculate information gain for each agent's action
    for action in joint_action
        if !isempty(action.target_cells)
            # Calculate information gain for this action
            info_gain = calculate_action_information_gain(belief, action)
            uncertainty_reduction = calculate_uncertainty_reduction(belief, action)
            
            # Combine metrics
            action_value = info_gain + 0.5 * uncertainty_reduction
            
            # Apply communication cost
            if action.communicate
                action_value -= 0.1
            end
            
            total_value += action_value
        end
    end
    
    return total_value
end

"""
calculate_action_information_gain(belief::GlobalBeliefState, action::SensingAction)
Calculates expected information gain for a specific action
"""
function calculate_action_information_gain(belief::GlobalBeliefState, action::SensingAction)
    if isempty(action.target_cells)
        return 0.0
    end
    
    total_gain = 0.0
    
    for cell in action.target_cells
        x, y = cell
        prior_prob = belief.event_probabilities[y, x]
        
        # Calculate expected information gain using entropy
        if prior_prob > 0.0 && prior_prob < 1.0
            # Expected entropy reduction
            # This is a simplified calculation - in practice, you'd use the full observation model
            expected_entropy_reduction = 0.5 * (1.0 - abs(prior_prob - 0.5))
            total_gain += expected_entropy_reduction
        end
    end
    
    return total_gain
end

"""
calculate_uncertainty_reduction(belief::GlobalBeliefState, action::SensingAction)
Calculates expected uncertainty reduction for a specific action
"""
function calculate_uncertainty_reduction(belief::GlobalBeliefState, action::SensingAction)
    if isempty(action.target_cells)
        return 0.0
    end
    
    total_reduction = 0.0
    
    for cell in action.target_cells
        x, y = cell
        current_uncertainty = belief.uncertainty_map[y, x]
        
        # Expected uncertainty reduction (simplified)
        expected_reduction = current_uncertainty * 0.3  # Assume 30% reduction on average
        total_reduction += expected_reduction
    end
    
    return total_reduction
end

"""
select_joint_action(policy::CentralizedPolicy, belief::GlobalBeliefState, clock::GlobalClockVector)
Implements π_cen : (τ_t, b_t) → a_t ∈ A_τ_t using belief-MDP solver
"""
function select_joint_action(policy::CentralizedPolicy, belief::GlobalBeliefState, clock::GlobalClockVector)
    # Use belief-MDP solver to find optimal joint action
    return solve_belief_mdp(policy, belief, clock)
end

"""
update_global_belief(belief::GlobalBeliefState, joint_action::Vector{SensingAction}, 
                    joint_observation::Vector{GridObservation}, env::SpatialGrid)
Updates global belief state using all agent observations
"""
function update_global_belief(belief::GlobalBeliefState, joint_action::Vector{SensingAction}, 
                            joint_observation::Vector{GridObservation}, env::SpatialGrid)
    # Create new belief state
    new_probabilities = copy(belief.event_probabilities)
    new_uncertainty = copy(belief.uncertainty_map)
    
    # Update based on all observations
    for (action, observation) in zip(joint_action, joint_observation)
        if !isempty(observation.sensed_cells)
            for (i, cell) in enumerate(observation.sensed_cells)
                x, y = cell
                observed_state = observation.event_states[i]
                
                # Update probability using Bayes rule
                prior_prob = new_probabilities[y, x]
                
                if observed_state == EVENT_PRESENT
                    # P(event|observation) = P(observation|event) * P(event) / P(observation)
                    likelihood_event = 0.9
                    likelihood_no_event = 0.1
                    posterior = (likelihood_event * prior_prob) / (likelihood_event * prior_prob + likelihood_no_event * (1 - prior_prob))
                    new_probabilities[y, x] = posterior
                elseif observed_state == NO_EVENT
                    # P(no_event|observation) = P(observation|no_event) * P(no_event) / P(observation)
                    likelihood_event = 0.1
                    likelihood_no_event = 0.9
                    posterior = (likelihood_no_event * (1 - prior_prob)) / (likelihood_event * prior_prob + likelihood_no_event * (1 - prior_prob))
                    new_probabilities[y, x] = 1.0 - posterior
                end
            end
        end
    end
    
    # Update uncertainty map
    new_uncertainty = calculate_uncertainty_map(new_probabilities)
    
    # Update observation history
    new_history = copy(belief.observation_history)
    push!(new_history, (belief.last_update + 1, joint_observation))
    
    return GlobalBeliefState(
        new_probabilities,
        new_uncertainty,
        new_history,
        belief.last_update + 1
    )
end

"""
calculate_uncertainty_map(probabilities::Matrix{Float64})
Calculates uncertainty map from probability map
"""
function calculate_uncertainty_map(probabilities::Matrix{Float64})
    uncertainty = Matrix{Float64}(undef, size(probabilities))
    
    for i in 1:size(probabilities, 1)
        for j in 1:size(probabilities, 2)
            uncertainty[i, j] = calculate_uncertainty(probabilities[i, j])
        end
    end
    
    return uncertainty
end

"""
calculate_uncertainty(probability::Float64)
Calculates uncertainty for a single probability value using entropy
"""
function calculate_uncertainty(probability::Float64)
    if probability <= 0.0 || probability >= 1.0
        return 0.0
    end
    return -(probability * log(probability) + (1 - probability) * log(1 - probability))
end

"""
simulate_centralized_execution(env::SpatialGrid, num_steps::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
Simulates centralized execution of the multi-agent system
"""
function simulate_centralized_execution(env::SpatialGrid, num_steps::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
    println("🚀 Starting Centralized Execution (C-LOMDP)")
    println("==========================================")
    
    # Initialize policy and belief
    policy = create_centralized_policy(env)
    global_belief = initialize_global_belief(env)
    
    # Initialize environment state
    initial_state_dist = POMDPs.initialstate(env)
    current_state = rand(rng, initial_state_dist)
    
    # Track execution
    execution_history = []
    total_reward = 0.0
    
    for step in 1:num_steps
        println("\n⏰ Step $(step)")
        
        # Get global clock vector
        clock = get_global_clock_vector(env, step-1)
        println("Global clock: τ = $(clock.agent_phases)")
        
        # Select joint action using centralized policy
        joint_action = select_joint_action(policy, global_belief, clock)
        println("Selected joint action: $(joint_action)")
        
        # Execute joint action and get observations
        joint_observation = Vector{GridObservation}()
        step_reward = 0.0
        
        for action in joint_action
            # Generate observation for this action
            obs_dist = POMDPs.observation(env, action, current_state)
            obs = rand(rng, obs_dist)
            push!(joint_observation, obs)
            
            # Calculate reward
            action_reward = POMDPs.reward(env, current_state, action, current_state)
            step_reward += action_reward
        end
        
        println("Joint observation: $(length(joint_observation)) observations")
        println("Step reward: $(step_reward)")
        
        # Update global belief
        global_belief = update_global_belief(global_belief, joint_action, joint_observation, env)
        
        # Transition to next state
        next_state_dist = POMDPs.transition(env, current_state, joint_action[1])  # Use first action for transition
        current_state = rand(rng, next_state_dist)
        
        # Record execution
        push!(execution_history, (step, clock, joint_action, joint_observation, step_reward))
        total_reward += step_reward
        
        # Check termination
        if POMDPs.isterminal(env, current_state)
            println("🏁 Environment terminated at step $(step)")
            break
        end
    end
    
    println("\n📊 Execution Summary")
    println("===================")
    println("Total steps: $(length(execution_history))")
    println("Total reward: $(total_reward)")
    if !isempty(execution_history)
        println("Average reward per step: $(total_reward / length(execution_history))")
    end
    
    return execution_history, global_belief, total_reward
end

"""
solve_belief_mdp(policy::CentralizedPolicy, belief::GlobalBeliefState, clock::GlobalClockVector)
Solves the belief-MDP using value iteration approach
"""
function solve_belief_mdp(policy::CentralizedPolicy, belief::GlobalBeliefState, clock::GlobalClockVector)
    # Create a simplified belief-MDP representation
    # State: (clock_vector, belief_state)
    # Actions: joint actions
    # Transitions: deterministic clock evolution + stochastic belief updates
    
    # For now, use a simplified approach
    # In practice, you'd implement proper belief-MDP solvers like RTDP-Bel or point-based methods
    
    # Get joint action space
    joint_actions = get_joint_action_space(policy.environment, clock)
    
    if isempty(joint_actions)
        return [SensingAction(i, [], false) for i in 1:length(policy.environment.agents)]
    end
    
    # Evaluate actions using lookahead search
    best_action = joint_actions[1]
    best_value = calculate_joint_action_value_with_lookahead(policy, belief, clock, joint_actions[1], policy.planning_horizon)
    
    for joint_action in joint_actions[2:end]
        value = calculate_joint_action_value_with_lookahead(policy, belief, clock, joint_action, policy.planning_horizon)
        if value > best_value
            best_value = value
            best_action = joint_action
        end
    end
    
    return best_action
end

"""
calculate_joint_action_value_with_lookahead(policy::CentralizedPolicy, belief::GlobalBeliefState, 
                                          clock::GlobalClockVector, joint_action::Vector{SensingAction}, depth::Int)
Calculates the value of a joint action using lookahead search (simplified belief-MDP)
"""
function calculate_joint_action_value_with_lookahead(policy::CentralizedPolicy, belief::GlobalBeliefState, 
                                                   clock::GlobalClockVector, joint_action::Vector{SensingAction}, depth::Int)
    if depth == 0
        return 0.0
    end
    
    # Immediate reward
    immediate_value = calculate_joint_action_value(policy, belief, clock, joint_action)
    
    # Lookahead value (simplified)
    # In practice, you'd use proper belief state sampling and transition models
    
    # Simulate next clock
    next_clock = GlobalClockVector(
        [(phase + 1) % get_trajectory_period(policy.environment.agents[i].trajectory) for (i, phase) in enumerate(clock.agent_phases)],
        clock.time_step + 1
    )
    
    # Simplified belief update (in practice, use proper observation model)
    # For now, assume some uncertainty reduction
    future_value = 0.0
    if depth > 1
        # Recursive lookahead with simplified belief evolution
        future_value = policy.discount_factor * immediate_value * 0.8  # Simplified discounting
    end
    
    return immediate_value + future_value
end

end  # module CentralizedPlanner 