#!/usr/bin/env julia

"""
Visualization script for the 2-state MA-LOMDP environment
Shows temporal evolution of events with animated grid visualizations
"""

using Pkg
Pkg.activate(".")

using POMDPs
using POMDPTools
using Random
using Plots

# Set random seed for reproducibility
Random.seed!(42)

# Define our own 2-state event system locally to avoid conflicts
@enum EventState2 begin
    NO_EVENT_2 = 0
    EVENT_PRESENT_2 = 1
end

# Define DBN transition model locally
struct DBNTransitionModel2
    birth_rate::Float64
    death_rate::Float64
    neighbor_influence::Float64
end

# Define belief state for proper DBN
struct BeliefState
    probabilities::Matrix{Float64}  # P(event_present) for each cell
    uncertainty::Matrix{Float64}    # Entropy/uncertainty for each cell
    time_step::Int
end

# Add copy method for BeliefState
Base.copy(belief::BeliefState) = BeliefState(
    copy(belief.probabilities),
    copy(belief.uncertainty),
    belief.time_step
)

# Define local functions for event dynamics
"""
Get neighbor states for a given position
"""
function get_neighbor_states(event_map::Matrix{EventState2}, x::Int, y::Int)
    neighbors = EventState2[]
    height, width = size(event_map)
    
    for dx in -1:1
        for dy in -1:1
            if dx == 0 && dy == 0
                continue
            end
            
            nx, ny = x + dx, y + dy
            if 1 <= nx <= width && 1 <= ny <= height
                push!(neighbors, event_map[ny, nx])
            end
        end
    end
    
    return neighbors
end

"""
Get neighbor beliefs for a given position
"""
function get_neighbor_beliefs(belief::BeliefState, x::Int, y::Int)
    neighbors = Float64[]
    height, width = size(belief.probabilities)
    
    for dx in -1:1
        for dy in -1:1
            if dx == 0 && dy == 0
                continue
            end
            
            nx, ny = x + dx, y + dy
            if 1 <= nx <= width && 1 <= ny <= height
                push!(neighbors, belief.probabilities[ny, nx])
            end
        end
    end
    
    return neighbors
end

"""
DBN local conditional probability for 2-state events
"""
function transition_probability_dbn(current_state::EventState2, neighbor_states::Vector{EventState2}, model::DBNTransitionModel2)
    num_active_neighbors = count(==(EVENT_PRESENT_2), neighbor_states)

    if current_state == EVENT_PRESENT_2
        # Persistence vs death
        return max(0.0, min(1.0, 1.0 - model.death_rate))
    else
        # Birth from neighbors
        return max(0.0, min(1.0, model.birth_rate + model.neighbor_influence * num_active_neighbors))
    end
end

"""
Proper DBN belief update using Bayesian inference
P(x_{t+1} = 1) = Σ_{x_t, neighbors} P(x_{t+1} = 1 | x_t, neighbors) * P(x_t, neighbors)
"""
function update_belief_dbn(belief::BeliefState, model::DBNTransitionModel2)
    height, width = size(belief.probabilities)
    new_probabilities = similar(belief.probabilities)
    new_uncertainty = similar(belief.uncertainty)
    
    for y in 1:height
        for x in 1:width
            # Get current belief and neighbor beliefs
            current_belief = belief.probabilities[y, x]
            neighbor_beliefs = get_neighbor_beliefs(belief, x, y)
            
            # Expected number of active neighbors
            E_neighbors = sum(neighbor_beliefs)
            
            # DBN belief update formula:
            # P(x_{t+1} = 1) = P(x_t = 1) * P(survive) + P(x_t = 0) * P(birth)
            p_survive = 1.0 - model.death_rate
            p_birth = model.birth_rate + model.neighbor_influence * E_neighbors
            
            new_prob = current_belief * p_survive + (1.0 - current_belief) * p_birth
            new_probabilities[y, x] = max(0.0, min(1.0, new_prob))
            
            # Calculate uncertainty (entropy)
            if new_prob > 0.0 && new_prob < 1.0
                new_uncertainty[y, x] = -(new_prob * log(new_prob) + (1.0 - new_prob) * log(1.0 - new_prob))
            else
                new_uncertainty[y, x] = 0.0
            end
        end
    end
    
    return BeliefState(new_probabilities, new_uncertainty, belief.time_step + 1)
end

"""
Sample state from belief (for visualization)
"""
function sample_from_belief(belief::BeliefState, rng::AbstractRNG)
    height, width = size(belief.probabilities)
    sampled_state = Matrix{EventState2}(undef, height, width)
    
    for y in 1:height
        for x in 1:width
            p = belief.probabilities[y, x]
            sampled_state[y, x] = rand(rng) < p ? EVENT_PRESENT_2 : NO_EVENT_2
        end
    end
    
    return sampled_state
end

"""
Initialize belief state
"""
function initialize_belief(height::Int, width::Int, initial_events::Int, rng::AbstractRNG)
    probabilities = fill(0.1, height, width)  # Low prior probability
    uncertainty = fill(0.5, height, width)    # High initial uncertainty
    
    # Add initial events
    for _ in 1:initial_events
        x = rand(rng, 1:width)
        y = rand(rng, 1:height)
        probabilities[y, x] = 0.9  # High probability for initial events
        uncertainty[y, x] = 0.1    # Low uncertainty for observed events
    end
    
    return BeliefState(probabilities, uncertainty, 0)
end

"""
Update events using DBN model (current implementation - more like cellular automaton)
"""
function update_events!(model::DBNTransitionModel2, event_map::Matrix{EventState2}, rng::AbstractRNG)
    height, width = size(event_map)
    new_map = similar(event_map)

    for y in 1:height
        for x in 1:width
            current = event_map[y, x]
            neighbors = get_neighbor_states(event_map, x, y)
            p = transition_probability_dbn(current, neighbors, model)
            new_map[y, x] = rand(rng) < p ? EVENT_PRESENT_2 : NO_EVENT_2
        end
    end

    event_map .= new_map
end

"""
Simulate environment evolution using cellular automaton approach (original)
"""
function simulate_environment_evolution_ca(width::Int, height::Int, num_steps::Int, initial_events::Int)
    println("🧠 Simulating 2-State Environment Evolution with Cellular Automaton")
    println("===================================================================")
    println("Grid size: $(width)x$(height)")
    println("Simulation steps: $(num_steps)")
    println("Initial events: $(initial_events)")
    
    # Create 2-state DBN model
    dbn_model = DBNTransitionModel2(0.001, 0.05, 0.01)
    println("DBN model: $(dbn_model)")
    
    # Initialize event map
    event_map = fill(NO_EVENT_2, height, width)
    
    # Add initial random events
    rng = Random.GLOBAL_RNG
    for _ in 1:initial_events
        x = rand(rng, 1:width)
        y = rand(rng, 1:height)
        event_map[y, x] = EVENT_PRESENT_2
    end
    
    println("Initial events: $(count(==(EVENT_PRESENT_2), event_map))")
    
    # Store evolution for animation
    evolution = [copy(event_map)]
    event_counts = [count(==(EVENT_PRESENT_2), event_map)]
    
    # Simulate evolution
    for step in 1:num_steps
        # Update events using DBN
        update_events!(dbn_model, event_map, rng)
        
        # Store current state
        push!(evolution, copy(event_map))
        push!(event_counts, count(==(EVENT_PRESENT_2), event_map))
        
        println("Step $(step): $(event_counts[end]) events")
    end
    
    return evolution, event_counts, dbn_model
end

"""
Simulate environment evolution using proper DBN
"""
function simulate_environment_evolution_dbn(width::Int, height::Int, num_steps::Int, initial_events::Int)
    println("🧠 Simulating 2-State Environment Evolution with Proper DBN")
    println("==========================================================")
    println("Grid size: $(width)x$(height)")
    println("Simulation steps: $(num_steps)")
    println("Initial events: $(initial_events)")
    
    # Create 2-state DBN model
    dbn_model = DBNTransitionModel2(0.001, 0.05, 0.01)
    println("DBN model: $(dbn_model)")
    
    # Initialize belief state
    rng = Random.GLOBAL_RNG
    belief = initialize_belief(height, width, initial_events, rng)
    
    # Sample initial state from belief
    event_map = sample_from_belief(belief, rng)
    
    println("Initial events: $(count(==(EVENT_PRESENT_2), event_map))")
    
    # Store evolution for animation
    evolution = [copy(event_map)]
    event_counts = [count(==(EVENT_PRESENT_2), event_map)]
    belief_evolution = [copy(belief)]
    
    # Simulate evolution using proper DBN
    for step in 1:num_steps
        # Update belief using DBN
        belief = update_belief_dbn(belief, dbn_model)
        
        # Sample state from updated belief
        event_map = sample_from_belief(belief, rng)
        
        # Store current state
        push!(evolution, copy(event_map))
        push!(event_counts, count(==(EVENT_PRESENT_2), event_map))
        push!(belief_evolution, copy(belief))
        
        println("Step $(step): $(event_counts[end]) events, avg belief: $(round(mean(belief.probabilities), digits=3))")
    end
    
    return evolution, event_counts, belief_evolution, dbn_model
end

"""
Compare DBN vs Cellular Automaton approaches
"""
function compare_approaches(width::Int, height::Int, num_steps::Int, initial_events::Int)
    println("🔬 Comparing DBN vs Cellular Automaton Approaches")
    println("=================================================")
    
    # Run both simulations
    println("\n1️⃣ Running Cellular Automaton simulation...")
    ca_evolution, ca_counts, ca_model = simulate_environment_evolution_ca(width, height, num_steps, initial_events)
    
    println("\n2️⃣ Running Proper DBN simulation...")
    dbn_evolution, dbn_counts, dbn_beliefs, dbn_model = simulate_environment_evolution_dbn(width, height, num_steps, initial_events)
    
    # Compare results
    println("\n📊 Comparison Results:")
    println("======================")
    println("Cellular Automaton:")
    println("- Final events: $(ca_counts[end])")
    println("- Max events: $(maximum(ca_counts))")
    println("- Min events: $(minimum(ca_counts))")
    println("- Avg events: $(round(mean(ca_counts), digits=2))")
    
    println("\nProper DBN:")
    println("- Final events: $(dbn_counts[end])")
    println("- Max events: $(maximum(dbn_counts))")
    println("- Min events: $(minimum(dbn_counts))")
    println("- Avg events: $(round(mean(dbn_counts), digits=2))")
    println("- Final avg belief: $(round(mean(dbn_beliefs[end].probabilities), digits=3))")
    
    # Create comparison plot
    p = plot(ca_counts, 
             label="Cellular Automaton",
             xlabel="Time Step",
             ylabel="Number of Events",
             title="DBN vs Cellular Automaton Comparison",
             linewidth=2,
             marker=:circle,
             markersize=3,
             grid=true)
    
    plot!(dbn_counts, 
          label="Proper DBN",
          linewidth=2,
          marker=:square,
          markersize=3)
    
    return p, ca_evolution, dbn_evolution, ca_counts, dbn_counts, dbn_beliefs
end

"""
Visualize a single grid state
"""
function visualize_grid(event_map::Matrix{EventState2}, title::String="Event Grid")
    height, width = size(event_map)
    
    # Create color matrix: 0 = white (no event), 1 = red (event present)
    color_matrix = zeros(Int, height, width)
    for y in 1:height
        for x in 1:width
            if event_map[y, x] == EVENT_PRESENT_2
                color_matrix[y, x] = 1
            end
        end
    end
    
    # Create heatmap
    p = heatmap(color_matrix, 
                aspect_ratio=:equal,
                color=:Reds,
                clim=(0, 1),
                title=title,
                xlabel="X",
                ylabel="Y",
                grid=false,
                showaxis=true,
                ticks=true)
    
    return p
end

"""
Create animation of environment evolution
"""
function create_environment_animation(evolution::Vector{Matrix{EventState2}}, event_counts::Vector{Int}, dbn_model::DBNTransitionModel2)
    println("\n🎬 Creating Environment Animation")
    println("================================")
    
    # Create frames for animation
    frames = []
    
    for (step, event_map) in enumerate(evolution)
        title = "Step $(step-1): $(event_counts[step]) events\nDBN: birth=$(dbn_model.birth_rate), death=$(dbn_model.death_rate), influence=$(dbn_model.neighbor_influence)"
        frame = visualize_grid(event_map, title)
        push!(frames, frame)
    end
    
    # Create animation
    anim = @animate for frame in frames
        plot(frame, size=(600, 600))
    end
    
    return anim
end

"""
Create event count plot over time
"""
function plot_event_evolution(event_counts::Vector{Int}, dbn_model::DBNTransitionModel2)
    println("📊 Creating Event Count Evolution Plot")
    println("=====================================")
    
    p = plot(event_counts, 
             label="Event Count",
             xlabel="Time Step",
             ylabel="Number of Events",
             title="Event Evolution Over Time\nDBN: birth=$(dbn_model.birth_rate), death=$(dbn_model.death_rate), influence=$(dbn_model.neighbor_influence)",
             linewidth=2,
             marker=:circle,
             markersize=3,
             grid=true)
    
    return p
end

"""
Create detailed analysis of environment evolution
"""
function analyze_environment_evolution(evolution::Vector{Matrix{EventState2}}, event_counts::Vector{Int})
    println("\n📈 Environment Evolution Analysis")
    println("================================")
    
    # Calculate statistics
    total_steps = length(event_counts)
    max_events = maximum(event_counts)
    min_events = minimum(event_counts)
    avg_events = mean(event_counts)
    
    println("Total simulation steps: $(total_steps)")
    println("Maximum events: $(max_events)")
    println("Minimum events: $(min_events)")
    println("Average events: $(round(avg_events, digits=2))")
    
    # Find peak and trough
    peak_step = argmax(event_counts)
    trough_step = argmin(event_counts)
    
    println("Peak events at step $(peak_step-1): $(event_counts[peak_step])")
    println("Minimum events at step $(trough_step-1): $(event_counts[trough_step])")
    
    # Calculate event spread patterns
    if length(evolution) > 1
        # Analyze spatial patterns
        println("\nSpatial Pattern Analysis:")
        
        # Get grid dimensions
        height, width = size(evolution[1])
        mid_height = div(height, 2)
        mid_width = div(width, 2)
        
        for step in [1, div(total_steps, 2), total_steps]
            event_map = evolution[step]
            
            # Count events in different regions (handle rectangular grids)
            q1 = count(x -> x == EVENT_PRESENT_2, event_map[1:mid_height, 1:mid_width])
            q2 = count(x -> x == EVENT_PRESENT_2, event_map[1:mid_height, mid_width+1:end])
            q3 = count(x -> x == EVENT_PRESENT_2, event_map[mid_height+1:end, 1:mid_width])
            q4 = count(x -> x == EVENT_PRESENT_2, event_map[mid_height+1:end, mid_width+1:end])
            
            println("Step $(step-1): Q1=$(q1), Q2=$(q2), Q3=$(q3), Q4=$(q4)")
        end
    end
end

"""
Visualize belief state (probability map)
"""
function visualize_belief(belief::BeliefState, title::String="Belief State")
    height, width = size(belief.probabilities)
    
    # Create heatmap of probabilities
    p = heatmap(belief.probabilities, 
                aspect_ratio=:equal,
                color=:Blues,
                clim=(0, 1),
                title=title,
                xlabel="X",
                ylabel="Y",
                grid=false,
                showaxis=true,
                ticks=true,
                colorbar_title="P(Event)")
    
    return p
end

"""
Visualize uncertainty map
"""
function visualize_uncertainty(belief::BeliefState, title::String="Uncertainty Map")
    height, width = size(belief.uncertainty)
    
    # Create heatmap of uncertainty (entropy)
    p = heatmap(belief.uncertainty, 
                aspect_ratio=:equal,
                color=:Reds,
                clim=(0, 0.7),  # Max entropy for binary is log(2) ≈ 0.693
                title=title,
                xlabel="X",
                ylabel="Y",
                grid=false,
                showaxis=true,
                ticks=true,
                colorbar_title="Entropy")
    
    return p
end

"""
Create animation of belief evolution
"""
function create_belief_animation(belief_evolution::Vector{BeliefState}, event_counts::Vector{Int}, dbn_model::DBNTransitionModel2)
    println("\n🎬 Creating Belief Evolution Animation")
    println("=====================================")
    
    # Create frames for animation
    frames = []
    
    for (step, belief) in enumerate(belief_evolution)
        title = "Step $(step-1): $(event_counts[step]) events\nDBN Belief: birth=$(dbn_model.birth_rate), death=$(dbn_model.death_rate), influence=$(dbn_model.neighbor_influence)"
        frame = visualize_belief(belief, title)
        push!(frames, frame)
    end
    
    # Create animation
    anim = @animate for frame in frames
        plot(frame, size=(600, 600))
    end
    
    return anim
end

"""
Create animation of uncertainty evolution
"""
function create_uncertainty_animation(belief_evolution::Vector{BeliefState}, event_counts::Vector{Int}, dbn_model::DBNTransitionModel2)
    println("\n🎬 Creating Uncertainty Evolution Animation")
    println("===========================================")
    
    # Create frames for animation
    frames = []
    
    for (step, belief) in enumerate(belief_evolution)
        title = "Step $(step-1): $(event_counts[step]) events\nDBN Uncertainty: birth=$(dbn_model.birth_rate), death=$(dbn_model.death_rate), influence=$(dbn_model.neighbor_influence)"
        frame = visualize_uncertainty(belief, title)
        push!(frames, frame)
    end
    
    # Create animation
    anim = @animate for frame in frames
        plot(frame, size=(600, 600))
    end
    
    return anim
end

"""
Analyze belief evolution
"""
function analyze_belief_evolution(belief_evolution::Vector{BeliefState})
    println("\n📈 Belief Evolution Analysis")
    println("============================")
    
    # Calculate statistics
    total_steps = length(belief_evolution)
    
    # Average belief over time
    avg_beliefs = [mean(belief.probabilities) for belief in belief_evolution]
    avg_uncertainties = [mean(belief.uncertainty) for belief in belief_evolution]
    
    println("Total simulation steps: $(total_steps)")
    println("Initial avg belief: $(round(avg_beliefs[1], digits=3))")
    println("Final avg belief: $(round(avg_beliefs[end], digits=3))")
    println("Initial avg uncertainty: $(round(avg_uncertainties[1], digits=3))")
    println("Final avg uncertainty: $(round(avg_uncertainties[end], digits=3))")
    
    # Find peak and trough in belief
    peak_belief_step = argmax(avg_beliefs)
    trough_belief_step = argmin(avg_beliefs)
    
    println("Peak belief at step $(peak_belief_step-1): $(round(avg_beliefs[peak_belief_step], digits=3))")
    println("Minimum belief at step $(trough_belief_step-1): $(round(avg_beliefs[trough_belief_step], digits=3))")
    
    return avg_beliefs, avg_uncertainties
end

"""
Main visualization function
"""
function main_visualization()
    println("🎨 2-State Environment Visualization")
    println("===================================")
    
    # Create output directory
    output_dir = "visualizations"
    if !isdir(output_dir)
        mkdir(output_dir)
        println("✓ Created output directory: $(output_dir)")
    end
    
    # Simulation parameters
    width = 6
    height = 30
    num_steps = 100
    initial_events = 1
    
    # Compare approaches
    comparison_plot, ca_evolution, dbn_evolution, ca_counts, dbn_counts, dbn_beliefs = compare_approaches(width, height, num_steps, initial_events)
    
    # Analyze evolution
    println("\n📈 Analysis of Proper DBN Evolution:")
    analyze_environment_evolution(dbn_evolution, dbn_counts)
    
    # Analyze belief evolution
    avg_beliefs, avg_uncertainties = analyze_belief_evolution(dbn_beliefs)
    
    # Create visualizations
    println("\n🎬 Creating Visualizations...")
    
    # Create animation for proper DBN
    dbn_model = DBNTransitionModel2(0.001, 0.05, 0.01)
    anim = create_environment_animation(dbn_evolution, dbn_counts, dbn_model)
    
    # Create belief animation
    belief_anim = create_belief_animation(dbn_beliefs, dbn_counts, dbn_model)
    
    # Create uncertainty animation
    uncertainty_anim = create_uncertainty_animation(dbn_beliefs, dbn_counts, dbn_model)
    
    # Create event count plot
    event_plot = plot_event_evolution(dbn_counts, dbn_model)
    
    # Create belief evolution plot
    belief_plot = plot(avg_beliefs, 
                      label="Average Belief",
                      xlabel="Time Step",
                      ylabel="P(Event)",
                      title="DBN Belief Evolution Over Time",
                      linewidth=2,
                      marker=:circle,
                      markersize=3,
                      grid=true)
    
    plot!(avg_uncertainties, 
          label="Average Uncertainty",
          linewidth=2,
          marker=:square,
          markersize=3,
          yaxis=:right)
    
    # Save results with configuration in filename
    println("\n💾 Saving Results...")
    
    # Create configuration string for filename
    config_str = "grid$(width)x$(height)_steps$(num_steps)_init$(initial_events)_birth$(dbn_model.birth_rate)_death$(dbn_model.death_rate)_influence$(dbn_model.neighbor_influence)"
    
    # Save animations as GIFs
    animation_filename = joinpath(output_dir, "environment_evolution_dbn_$(config_str).gif")
    gif(anim, animation_filename, fps=2)
    println("✓ DBN Animation saved as '$(animation_filename)'")
    
    belief_animation_filename = joinpath(output_dir, "belief_evolution_dbn_$(config_str).gif")
    gif(belief_anim, belief_animation_filename, fps=2)
    println("✓ Belief Animation saved as '$(belief_animation_filename)'")
    
    uncertainty_animation_filename = joinpath(output_dir, "uncertainty_evolution_dbn_$(config_str).gif")
    gif(uncertainty_anim, uncertainty_animation_filename, fps=2)
    println("✓ Uncertainty Animation saved as '$(uncertainty_animation_filename)'")
    
    # Save plots
    plot_filename = joinpath(output_dir, "event_evolution_dbn_$(config_str).png")
    savefig(event_plot, plot_filename)
    println("✓ DBN Event evolution plot saved as '$(plot_filename)'")
    
    belief_plot_filename = joinpath(output_dir, "belief_evolution_dbn_$(config_str).png")
    savefig(belief_plot, belief_plot_filename)
    println("✓ Belief evolution plot saved as '$(belief_plot_filename)'")
    
    # Save comparison plot
    comparison_filename = joinpath(output_dir, "dbn_vs_ca_comparison_$(config_str).png")
    savefig(comparison_plot, comparison_filename)
    println("✓ Comparison plot saved as '$(comparison_filename)'")
    
    # Display final states
    println("\n🎯 Final Environment State (Proper DBN):")
    final_state = visualize_grid(dbn_evolution[end], "Final State: $(dbn_counts[end]) events")
    display(final_state)
    
    println("\n🎯 Final Belief State:")
    final_belief = visualize_belief(dbn_beliefs[end], "Final Belief: avg=$(round(mean(dbn_beliefs[end].probabilities), digits=3))")
    display(final_belief)
    
    println("\n🎯 Final Uncertainty State:")
    final_uncertainty = visualize_uncertainty(dbn_beliefs[end], "Final Uncertainty: avg=$(round(mean(dbn_beliefs[end].uncertainty), digits=3))")
    display(final_uncertainty)
    
    println("\n✅ Visualization completed!")
    println("\n📁 Generated files in '$(output_dir)' folder:")
    println("- $(basename(animation_filename)): DBN animated grid evolution")
    println("- $(basename(belief_animation_filename)): DBN belief evolution")
    println("- $(basename(uncertainty_animation_filename)): DBN uncertainty evolution")
    println("- $(basename(plot_filename)): DBN event count over time")
    println("- $(basename(belief_plot_filename)): DBN belief evolution over time")
    println("- $(basename(comparison_filename)): DBN vs Cellular Automaton comparison")
    
    return anim, belief_anim, uncertainty_anim, event_plot, belief_plot, comparison_plot, final_state, final_belief, final_uncertainty
end

"""
Interactive visualization with different parameters
"""
function interactive_visualization()
    println("🎮 Interactive Environment Visualization")
    println("=======================================")
    
    # Create output directory
    output_dir = "visualizations"
    if !isdir(output_dir)
        mkdir(output_dir)
        println("✓ Created output directory: $(output_dir)")
    end
    
    # Test different parameter combinations
    parameter_sets = [
        ("High_Birth_Rate", DBNTransitionModel2(0.01, 0.05, 0.1)),
        ("High_Death_Rate", DBNTransitionModel2(0.005, 0.1, 0.01)),
        ("High_Neighbor_Influence", DBNTransitionModel2(0.01, 0.05, 0.3)),
        ("Balanced", DBNTransitionModel2(0.01, 0.01, 0.01))
    ]
    
    for (name, dbn_model) in parameter_sets
        println("\n🔬 Testing: $(name)")
        println("Parameters: birth=$(dbn_model.birth_rate), death=$(dbn_model.death_rate), influence=$(dbn_model.neighbor_influence)")
        
        # Simulate with these parameters
        width = 10
        height = 10
        num_steps = 10
        initial_events = 1
        
        # Initialize and simulate
        event_map = fill(NO_EVENT_2, height, width)
        event_map[4, 4] = EVENT_PRESENT_2  # Start with one event in center
        
        evolution = [copy(event_map)]
        event_counts = [1]
        
        rng = Random.GLOBAL_RNG
        for step in 1:num_steps
            update_events!(dbn_model, event_map, rng)
            push!(evolution, copy(event_map))
            push!(event_counts, count(==(EVENT_PRESENT_2), event_map))
        end
        
        # Create configuration string for filename
        config_str = "grid$(width)x$(height)_steps$(num_steps)_init$(initial_events)_birth$(dbn_model.birth_rate)_death$(dbn_model.death_rate)_influence$(dbn_model.neighbor_influence)"
        
        # Create animation for this parameter set
        anim = create_environment_animation(evolution, event_counts, dbn_model)
        animation_filename = joinpath(output_dir, "environment_$(name)_$(config_str).gif")
        gif(anim, animation_filename, fps=2)
        
        println("✓ Saved animation: $(basename(animation_filename))")
    end
    
    println("\n📁 All animations saved in '$(output_dir)' folder")
end

# Run the main visualization
if abspath(PROGRAM_FILE) == @__FILE__
    main_visualization()
    
    # Uncomment to run interactive visualization
    # interactive_visualization()
end 