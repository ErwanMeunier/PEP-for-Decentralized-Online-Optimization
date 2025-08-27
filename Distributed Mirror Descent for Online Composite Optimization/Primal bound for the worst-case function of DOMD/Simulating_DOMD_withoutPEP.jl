using Symbolics
using LinearAlgebra
using Plots
using ColorSchemes
using Random

# For solving the minimization subproblem
using JuMP
using Ipopt

# Multi-threading
using Base.Threads

###
Random.seed!(1)

include("generating_doubly_stochastic_matrices.jl")

function declaring_functions(n,T,L_ref=1,R_ref=1)
    Symbolics.@variables x, y 
    # Functions of agents
    f = Matrix{Num}(undef, n,T)
    minima_fit = Matrix{Tuple{Float64,Float64}}(undef,n,T)
    for i=1:n
        for t=1:T
            f[i,t] = L_ref*((x - (i-1)/(n*sqrt(2)))^2 + (y - (t-1)/(T*sqrt(2)))^2)/(2*2*R_ref)
            minima_fit[i,t] = ((i-1)/(n*sqrt(2)),(t-1)/(T*sqrt(2)))
        end 
    end
    #Dx = Differential(x) # Instanciating the differential operator
    #Dy = Differential(y)
    #∇f = collect(map(func -> expand_derivatives.([Dx(func),Dy(func)]),f)) # Computing the derivatives
    ∇f = collect(map(func -> Symbolics.gradient(func,[x,y]),f))
    # Transforming expression to functions
    f_fun = Matrix{Any}(undef,n,T)
    ∇f_fun = Matrix{Any}(undef,n,T)
    for i=1:n
        for t=1:T
            f_fun[i,t] = (xval,yval) -> eval(substitute(f[i,t],Dict(x=>xval,y=>yval)))
            # evaluating along side each dimension 
            ∇f_fun[i,t] = (xval,yval) -> [eval(substitute(∇f[i,t][k],Dict(x=>xval,y=>yval))) for k=1:2]
        end 
    end
    # Kernel for the Bregman Divergence
    ω(x) = norm(x)^2/2
    ∇ω(x) = x
    # Bregman Divergence
    V(x,y) = ω(x) - ω(y) - (∇ω(y)')*(x-y)
    return f_fun, ∇f_fun, ω, ∇ω, V, minima_fit
end

###########################################
#  Crafting random functions / sampling   #
###########################################

function r(R,A,N,Nsamp,i)
    return sqrt(R^2-Nsamp*A*(N-i)/(pi))
end

# Nsamp is for angles --> Number of sectors
# N concentric circles --> Number of radii
function drawing_uniform_discretization(R,N,Nsamp;plotting=true)
    A = R^2 * pi / (Nsamp * N)
    # Sampling angles
    θ_samp = range(0,2*pi,Nsamp+1)[1:end]
    # Sampling radii
    r_samp = collect(map(i->r(R,A,N,Nsamp,i),1:N))
    # --> Circles
    range_theta_for_plotting = collect(range(0,2pi,1000))
    # Computing point in the middle of the region
    W = Matrix{Tuple{Float64,Float64}}(undef,Nsamp,N)
    for i in 1:Nsamp
        for j in 1:N
            angle_w = i > 1 ? (θ_samp[i]+θ_samp[i+1])/2 : θ_samp[i+1]/2
            radius_w = j > 1 ? (r_samp[j-1] + r_samp[j])/2 : (r_samp[j]/2)
            W[i,j] = radius_w .* (cos(angle_w),sin(angle_w))
        end
    end
    # Plotting
    if plotting
        fig = plot(title="Discretized Disc | N="*string(Nsamp*N)*" tesselations",legend=nothing,aspect_ratio = :equal,size=(500,500),dpi=100) 
        # --> Slices 
        for θ in θ_samp 
            plot!(fig,[0,R*cos(θ)],[0,R*sin(θ)],color="blue")
        end
        for r in r_samp
            plot!(fig,r .* cos.(range_theta_for_plotting),r .* sin.(range_theta_for_plotting) ,color="blue") 
        end
        list_points = [w for w in W]
        scatter!(fig,first.(list_points),last.(list_points),color="red",markersize=1)
        return fig, θ_samp, r_samp, W
    else 
        return θ_samp, r_samp, W
    end
end

# Animating the sampling (just for fun)
function sampling_to_gif()
    anim = @animate for i in 2:90
        println(i)
        drawing_uniform_discretization(1,i,i;plotting=true)
    end

    gif(anim, "my_animation.gif", fps=10)
end

# The number of samples for the sectors and the radii
# T: Number of Iterations
# n: Number of agents
# minima_fit_arg can be provided so to avoid recomputing it every time
function sampling_functions(T,n,R=1;NbSamplesSectors=100,NbSamplesRadii=100,sampling_positions_arg=nothing,L_ref=nothing)
    W = isnothing(sampling_positions_arg) ? drawing_uniform_discretization(R,NbSamplesRadii,NbSamplesSectors;plotting=false)[3] : sampling_positions_arg
    #@show isnothing(sampling_positions_arg)
    #minima_fit = Matrix{Tuple{Float64,Float64}}(n,T)
    size_ax1 = size(W,1)
    size_ax2 = size(W,2)
    minima_fit = Matrix{Tuple{Float64,Float64}}(undef,n,T)
    for i=1:n 
        for t=1:T
            minima_fit[i,t] = W[rand(1:size_ax1),rand(1:size_ax2)]
        end
    end
    ###################################################
    Symbolics.@variables x, y 
    # Functions of agents
    f = Matrix{Num}(undef, n,T)
    for i=1:n
        for t=1:T
            f[i,t] = (x - minima_fit[i,t][1])^2 + (y - minima_fit[i,t][2])^2 
            if !isnothing(L_ref) # If L_ref has a value f[i,t] is "normalized" and scaled up with L_ref
                f[i,t] = L_ref * f[i,t] / 2*(norm(minima_fit,2)+R)
            end
        end 
    end
    
    ∇f = collect(map(func -> Symbolics.gradient(func,[x,y]),f))
    # Transforming expression to functions
    f_fun = Matrix{Any}(undef,n,T)
    ∇f_fun = Matrix{Any}(undef,n,T)
    for i=1:n
        for t=1:T
            f_fun[i,t] = (xval,yval) -> eval(substitute(f[i,t],Dict(x=>xval,y=>yval)))
            # evaluating along side each dimension 
            ∇f_fun[i,t] = (xval,yval) -> [eval(substitute(∇f[i,t][k],Dict(x=>xval,y=>yval))) for k=1:2]
        end 
    end
    # Kernel for the Bregman Divergence
    ω(x) = norm(x)^2/2
    ∇ω(x) = x
    # Bregman Divergence
    V(x,y) = ω(x) - ω(y) - (∇ω(y)')*(x-y)
    return f_fun, ∇f_fun, ω, ∇ω, V, minima_fit
end

##############################################################
#                         Other tools                        #
##############################################################

#= Compute the offline total cost (and the minimizer coming along)
 The global offline minimizer is assumed to be inside the feasible set
 + minima_fit : Matrix{Tuple{Float64,Float64}}
=#
function offline_totalcost(n,T,f_fun;random_functions=true,minima_fit::Union{Matrix{Tuple{Float64,Float64}},Nothing}=nothing,L_ref=nothing,R_ref=nothing)
    if random_functions
        x_min = zeros(Float64,2)
        if isnothing(L_ref)
            # x_min= Average of the minimizers is the minimizer
            for t=1:T
                for i=1:n
                    x_min = x_min .+ minima_fit[i,t]
                end 
            end 
            x_min = x_min ./ (n*T)
        else # Since functions were rescaled, they have distincts weights thus making the global offline minimizer no longer the average of local offline minimizers
            denominator = sum([sum([1/(norm(minima_fit[i,t],2)+R_ref) for i=1:n]) for t=1:T]) # scaling term
            x_min[1] = sum([sum([minima_fit[i,t][1]/(norm(minima_fit,2)+R_ref) for i=1:n]) for t=1:T]) / denominator
            x_min[2] = sum([sum([minima_fit[i,t][2]/(norm(minima_fit,2)+R_ref) for i=1:n]) for t=1:T]) / denominator
        end
        # Computing the total offline loss:
        return sum([sum([f_fun[i,t](x_min...) for i=1:n]) for t=1:T]) 
    else 
        if !isnothing(L_ref) && !isnothing(R_ref) 
            x_min = 2*R_ref*(1-1/(T*n))/(2*sqrt(2)) .* (1,1)
            return (L_ref/4*R_ref)*sum([sum([f_fun[i,t](x_min...) for i=1:n]) for t=1:T]) 
        else
            #@show "Last case"
            return T*n*(n^2+T^2-2)/12 # should not be used
        end
    end
end


# Computing the radius such that all minimizers belong to the feasible set
function computing_R_global(n,T;R_ref=nothing,random_functions=true)
    if random_functions || !isnothing(R_ref)
        return R_ref
    else
        return norm(1/2*[n+1,T+1],2) 
    end
end

# Computing the biggest Lipschitz constant for w.r.t the way the set of functions is defined (static or random)
function computing_Lipschitz_constant(n,T,minima_fit,R;random_functions=true)
    if random_functions
        arg = argmax(map(x->norm(x,2),minima_fit))
        i, t = arg[1], arg[2]
        x_max_gradient_magnitude = R .* (.- minima_fit[i,t] ./ norm(minima_fit[i,t],2)) # The maximum magnitude for the gradient is clearly on the border of the ball
        return norm(2 .* ((x_max_gradient_magnitude[1] .- minima_fit[i,t]) .+ (x_max_gradient_magnitude[2] .- minima_fit[i,t])),2)
    else
        return T*n*sqrt((9/4)*(n^2+T^2) + (3/2)*(n+T) + 1/4)
    end
end

##############################################################
#                         Algorithm                         #
##############################################################

# To be used in DOMD  
function compute_argmin_ball(xit,∇it,η,R)
    x = xit .- (η * ∇it)
    norm_x = norm(x,2)
    if  norm_x <= R
        return x
    else
        return (x ./ (norm(x,2)-R*(2*η+1))) .* R 
    end
end

# Here, R is typically D/2 (since we use a diameter bounded feasible set).
function compute_argmin_rectangle(xit,∇it,η,R)
    x = xit .- (η * ∇it)
    # checking whether this solution lies in the feasible set; i.e. 0 <= x <= sqrt(2)*R 
    if  (0<= x[1] <= sqrt(2)*R) && (0 <= x[2] <= sqrt(2)*R)
        #@show x
        return x
    else # we use IPOPT to solve the (convex) subproblem
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        @variable(model, 0 <= w[1:2] <= R*sqrt(2))
        @objective(model, Min, (w[1]^2 + w[2]^2)/(2*η) + w' *  (Float64.(Symbolics.value.(∇it)) - (Float64.(Symbolics.value.(xit)) ./ η)))
        optimize!(model)
        #@show value.(w)
        return value.(w)
    end
end

function DOMD(n,η,R,P,∇f,∇ω,T,random_functions)
    x = Matrix{Any}(undef,n,T+1) # Estimate
    y = Matrix{Any}(undef,n,T)
    ∇f_val = Matrix{Any}(undef,n,T) # Local gradient value
    ∇ω_val = Matrix{Any}(undef,n,T) # Local gradient kernel function
    for i=1:n
        x[i,1] = [0.,0.] # Minimizer of the Kernel ω(x) = ||x||^2
    end
    for t=1:T
        for i=1:n
            # Evaluating Fist-order Operators
            ∇f_val[i,t] = ∇f[i,t](x[i,t]...) 
            ∇ω_val[i,t] = ∇ω(x[i,t])
            # random_functions == true => ball centered in 0 with radius R; otherwise hyperectangle 0 <= x <= D*sqrt(2)/ 2 = R*sqrt(2)
            y[i,t] = random_functions ? 
                            compute_argmin_ball(x[i,t],∇f_val[i,t],η,R) :
                            compute_argmin_rectangle(x[i,t],∇f_val[i,t],η,R)
        end
        # mixing step
        for i=1:n
            x[i,t+1] = sum(P[i,j]*y[j,t] for j=1:n)
        end
    end
    #display(x)
    return x
end

##############################################################
#                         Benchmarking                       #
##############################################################

function run_DOMD(η,P,R,T;random_functions=true,sampling_positions_arg=nothing,L_ref=nothing)
    n = size(P,1) # number of agents 
    minima_fit = random_functions ? Matrix{Tuple{Float64,Float64}}(undef,n,T) : nothing
    f_fun, ∇f_fun, _, ∇ω, _, minima_fit = random_functions ? sampling_functions(T,n,R;sampling_positions_arg=sampling_positions_arg,L_ref=L_ref) : declaring_functions(n,T)
    x = DOMD(n,η,R,P,∇f_fun,∇ω,T,random_functions)
    # Computing the regret 
    # Computing the minimizer of the functions 
    ONLINE_PERF = maximum([sum([sum(f_fun[i,t](x[j,t]...) for i=1:n) for t=1:T]) for j=1:n];init=-Inf64)
    #@show ONLINE_PERF
    OFFLINE_PERF = offline_totalcost(n,T,f_fun;random_functions=random_functions,minima_fit=minima_fit,L_ref=L_ref,R_ref=R)
    #@show OFFLINE_PERF
    return Symbolics.value.(ONLINE_PERF - OFFLINE_PERF), minima_fit
end


#= W can be passed as argument so to avoid recomputing it each time --> see sampling_positions_arg
--> Computing and plotting the scaled regret for a network ranging from n_low to n_high agents given: 
 + T_global : number of iterations allowed for DOMD 
 + R_ref : radius ;
 + random_functions : parameters telling whether functions are generated on a n_high x T_global grid or Randomly
                      in the feasible set spanned by R_ref
 + sampling_positions_arg : MUST BE NOT NOTHING if random_functions == true. Gives the position of the minimizers 
                            (thus spanning the every f_{i,t}).
 + L_ref : If any value is given to L_ref, then all functions will be made L_ref Lipschitz. FOR NOW CAN BE USED ONLY IF RANDOM_FUNCTIONS==true
=#
function plot_wrt_N(n_low, n_high, T_global = 15; 
                    random_functions=true, sampling_positions_arg=nothing,  
                    R_ref=1, L_ref=nothing)

    R_global = computing_R_global(n_high, T_global; R_ref=R_ref, random_functions) 

    bench = (n,T) -> run_DOMD(1/sqrt(T), zeros(n,n) .+ 1/n, R_global, T_global;
                              random_functions=random_functions,
                              sampling_positions_arg=sampling_positions_arg,
                              L_ref=L_ref)
    #=bench = (n,T) -> run_DOMD(1/sqrt(T), star_graph(n), R_global, T_global;
                              random_functions=random_functions,
                              sampling_positions_arg=sampling_positions_arg,
                              L_ref=L_ref)=#

    bench_wrt_N = (n) -> bench(n, T_global)

    N_range = collect(n_low:n_high)
    Scaled_Regret = Vector{Float64}(undef, length(N_range))

    @threads for i in eachindex(N_range)
        n = N_range[i]
        println("Thread $(threadid()) working on N = $n / $(N_range[end])")
        regret, minima_fit = bench_wrt_N(n)
        L = isnothing(L_ref) ? computing_Lipschitz_constant(n, T_global, minima_fit, R_ref;
                                                            random_functions=random_functions) : L_ref
        Scaled_Regret[i] = regret / (2 * R_global * T_global * n * L)
    end

    fig = scatter(N_range, Scaled_Regret, 
                  title="Scaled Regret w.r.t. Number of Agents for DOMD", linewidth=3)
    ylabel!(fig, "Scaled Regret")
    xlabel!(fig, "N: Number of Agents")
    
    return Scaled_Regret, fig
end


##########################################
#  BENCHMARKING FOR VARIOUS VALUES OF T  #
##########################################
function main(; T_LOW = 15, STEP_T=1, T_HIGH = 15, N_LOW = 2, N_HIGH = 50, repetitions_arg=1, random_functions=true, L_ref=1, R_ref=0.5)
    repetitions = random_functions ? repetitions_arg : 1
    fig_sampling, _, _, sampling_positions_arg = drawing_uniform_discretization(1,10,10;plotting=true)
    Results_benchmark = Matrix{Vector{Float64}}(undef,length(T_LOW:T_HIGH),repetitions)
    fig_output = plot(title="Lower-bound over ISR for DOMD",titlefont=(14,"Computer Modern"),tickfontsize=10,legend=:outerright;palette=:viridis,size=(500,325))
    T_range = T_LOW:STEP_T:T_HIGH
    palette_sampled = T_LOW==T_HIGH ? 
                            [ColorSchemes.viridis[70]] : # Handling the case where length(T_range)==1 otherwise LinRange throws an error
                            get(ColorSchemes.viridis, LinRange(0, 1, length(T_range)) )
    for i in eachindex(T_range)
        println("T=",T_range[i])
        for rep = 1:repetitions
            println("--> Rep=",rep)
            Results_benchmark[i,rep],_ = plot_wrt_N(N_LOW,N_HIGH,T_range[i];random_functions=random_functions,sampling_positions_arg=sampling_positions_arg, L_ref=L_ref, R_ref=R_ref)
        end
        averaged_output = Results_benchmark[i,1]
        for rep = 2:repetitions
            averaged_output .+= Results_benchmark[i,rep]
        end 
        averaged_output = averaged_output ./ repetitions
        # Handling the case where length(T_range)==1
        fig_output = plot!(fig_output,N_LOW:N_HIGH,averaged_output,
                            label=string(T_range[i]),
                            palette=palette_sampled,
                            xrotation=75,
                            linewidth=2,
                            legendtitle="T",
                            xlims=(N_LOW,N_HIGH),
                            #ylims=(0,0.101)
                            )  
    end
    xlabel!(fig_output,L"$N$")
    ylabel!(fig_output,L"$\overline{\mathbf{Reg}}_j(T)$")
    xticks!(fig_output, [[N_LOW];5:5:N_HIGH])
    savefig(fig_output, "figure__DOMD_simulations.svg")
    combined_fig = random_functions ? plot(fig_output,fig_sampling,layout=(1,2),size=(600,600),dpi=300) : fig_output
    savefig(combined_fig,"./combined_fig.png")
    return Results_benchmark, combined_fig
end


##########################################
#              Graph generators           #
##########################################

# first agent is supposed to be the center of the graph
function star_graph(n) # OK
    W = Matrix{Float64}(undef,n,n)
    for i=1:n
        for j=1:n
            if (i== 1 || j==1)
                W[i,j] = 1/n
            else
                if i==j
                    W[i,j]=1-1/n
                else
                    W[i,j] = 0
                end
            end
        end
    end
    W[1,1]=1/n
    return W
end

#= Interesting instances =#
# results, fig = main(;T_LOW=5, STEP_T=5,T_HIGH=25, N_LOW=2, N_HIGH=20, random_functions=false, L_ref=1, R_ref=0.5)