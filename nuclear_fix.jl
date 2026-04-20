#!/usr/bin/env julia

println("🚀 Nuclear option: Complete package environment reset...")

using Pkg
Pkg.activate(".")

println("1. Removing ALL packages from Project.toml...")
# Read current Project.toml
project_content = read("Project.toml", String)
lines = split(project_content, '\n')

# Find the [deps] section and remove all packages except basic ones
new_lines = String[]
in_deps = false
basic_packages = ["Dates", "Random", "Statistics", "LinearAlgebra"]

for line in lines
    if line == "[deps]"
        in_deps = true
        push!(new_lines, line)
    elseif line == "" && in_deps
        in_deps = false
        push!(new_lines, line)
    elseif in_deps && startswith(line, "[")
        in_deps = false
        push!(new_lines, line)
    elseif in_deps
        # Check if it's a basic package
        package_name = split(line, " = ")[1]
        if package_name in basic_packages
            push!(new_lines, line)
        end
        # Skip all other packages
    else
        push!(new_lines, line)
    end
end

# Write the cleaned Project.toml
write("Project.toml", join(new_lines, '\n'))

println("2. Removing Manifest.toml...")
if isfile("Manifest.toml")
    rm("Manifest.toml")
end

println("3. Adding essential packages...")
Pkg.add(["CSV", "DataFrames", "Plots"])

println("4. Resolving and instantiating...")
Pkg.resolve()
Pkg.instantiate()

println("✅ Nuclear reset complete!")


