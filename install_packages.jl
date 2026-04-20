#!/usr/bin/env julia

using Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

println("Packages installed successfully!")