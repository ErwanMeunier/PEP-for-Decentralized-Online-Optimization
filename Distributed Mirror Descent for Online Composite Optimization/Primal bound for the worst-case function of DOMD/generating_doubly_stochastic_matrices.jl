using JuMP
using HiGHS

function doubly_stochastic_matrix(n, K)

    P_sampling = Vector{Matrix}(undef, K)
    for k=1:K
        model = Model(HiGHS.Optimizer)
        set_silent(model)

        @variable(model, 0 <= P[1:n,1:n] <= 1)
        
        # Random objective function aimed at diversifying returned solutions
        w = rand(n,n)
        @objective(model, Max, sum(w[i,j]*P[i,j] for i=1:n for j=1:n))
        
        @constraint(model, P' == P)
        @constraint(model, P*ones(n) == ones(n))

        optimize!(model)
        P_sampling[k] = value.(P)
    end
    return P_sampling
end

function convex_mixing(P_sampling)
    K = length(P_sampling)
    w = rand(K)
    w ./= sum(w)
    return sum(P_sampling[k]*w[k] for k=1:K)
end

# Return a random doubly stochastic symmetric matrix
# N is the number of random matrices to be returned
# K: Sampling size (~100)
function rand_DS_Sym(n,K,N)
    PS = doubly_stochastic_matrix(n, K)
    return [convex_mixing(PS) for _=1:N]
end