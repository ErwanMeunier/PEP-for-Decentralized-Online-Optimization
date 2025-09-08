% Function which returns the TIGHTEST bound for the Distributed Online
% Mirror Descent introduced by Based on Yuan, D., Hong, Y., Ho, D. W., 
% & Xu, S. (2020). Distributed mirror descent for online composite optimization.
% IEEE Transactions on Automatic Control, 66(2), 714-729.

% Input:
% L: Lipschitz constant of the loss functions
% G: Lipschitz constant on the gradient of the kernel function w <->
% Smoothness
% Lambda: Strong-convexity parameter of the Kernel Function
% n: Number of agents
% verbose: 0 --> No verbose, 1 --> Full verbose
% sigma: Second largest eigenvalue
function [wc,sum_norm_first_estimates,sum_Bregman_first_estimates,eta,min_val]=...
    distributed_mirror_descent_online_optimization(T,D,L,G,lambda,n,performance_metric,verbose,sigma,THRESHOLD_MIN_VAL_DOMD)

    % We define a threshold for the constraint to be satisfied. Otherwise
    % we get numerical problems with the solver
    % THRESHOLD_HARD_CONSTRAINT = 10^(-6);

    % Setting the network topology ----------------------------------------
    type = 'spectral_relaxed'; % The topology is defined by the spectrum of Consensus matrix 
    time_varying_mat = 0; % The communication network is supposed to be dynamic
    mat = [-sigma,sigma]; % range of eigenvalues for the communication matrix
    returnOpt = 0; 
    % equalStart = 1; % Initial estimates minimize the kernel function w
    
    % Setting-up step-sizes -----------------------------------
    c = 1;
    eta = c / sqrt(T);

    % Initialize an empty PEP
    P = pep();
    
    % Set up the local and global objective functions -----------------
    fctClass = 'ConvexBoundedGradient'; % Class of functions to consider for the worst-case
    fctParam.R = L; % Bounded Gradient constant ||g||^2 <= L^2.  

    FI = cell(n, T+1); % A matrix of mxT functions

    for t=1:T+1
        [Fi,~,~,~] = P.DeclareMultiFunctions(fctClass,fctParam,n,returnOpt);
        FI(:,t) = Fi;
    end

    kernelClass = 'SmoothStronglyConvex';
    kernelParam.L = G; % Setting the smoothness of the Kernel
    kernelParam.mu = lambda; % Setting the strong-convexity of the Kernel
    w = P.DeclareFunction(kernelClass,kernelParam); % Kernel of the Bregman Method

    % BOUNDED DOMAIN VIA INDICATOR FUNCTION
    param_id.D = D; % Diameter of the ball defining the compact set: ||x-y||^2 <= D^2, for all x and y
    id_function = P.DeclareFunction('ConvexIndicator',param_id);
    
    % Getting the optimal point and value to compute the regret in due time
    F_sum_with_id = id_function; 
    for t=1:T
        F_sum_with_id = F_sum_with_id + sumcell(FI(:,t));
    end
    % Defining the optimal point to consider the regret as the performance metric 
    [u, fs] = F_sum_with_id.OptimalPoint();

    % Iterates cells 
    Y = cell(n, T);          
    X = cell(n, T+1);      
   
    %F_saved = cell(m,T+1); NO NEED TO SAVE IT
    G_saved = cell(n,T+1);

    % Starting points must be minimizers of w over the bounded set
    w_id = w + id_function; 
    xstar_w_id = w_id.OptimalPoint(); 
   
    % Set up the starting points and initial conditions -------------------
    
    for i=1:n % All starting points are equal to the minimizer of the kernel function
        X(i,1) = {xstar_w_id};
    end

    % Then, all X(i,1) are minimizers of the kernel w over the bounded domain.

    [G_saved(:,1),~] = LocalOracles(FI(:,1),X(:,1)); % F_1(X(1,1)), F_2(X(2,1)), ...

    % Set up the averaging matrix -----------------------------------------
    A = P.DeclareConsensusMatrix(type,mat,time_varying_mat);
    % Algorithm -----------------------------------------------------
    % Iterations 
    for t = 1:T
        for i=1:n
            % --- Computing the mirror step with the Bregman Divergence
            [gw_xit,~,~] = w.oracle(X{i,t}); % Computing g(w(x_it))
            y = Point('Point'); % This point is the new point obtained after the mirror step
            [gw_y,~,~] = w.oracle(y); % Computing g(w(y))
            [gid_y,~,~] = id_function.oracle(y); % Computing g(id(y))
            % Step defined by the following constraint
            P.AddConstraint((G_saved{i,t}+(1/eta)*(gw_y-gw_xit))^2==0);
           
            % ---
            Y(i,t) = {y}; % Estimates before communicating
        end
        AxY = A.consensus(Y(:,t));
        for i=1:n
            X(i,t+1) = AxY(i); % Communicating with its current neighbors
        end
        [G_saved(:,t+1),~] = LocalOracles(FI(:,t+1),X(:,t+1)); % --> We could save F as well 
    end
    
    % SETTING THE PERFORMANCE METRIC --------------------------------------
    fixed_agent_j = 1; % The regret is the same regardless the agent.
    F_sum_fixed = 0;
    F_saved_fixed_agent = cell(n,T); % Each cell contains F_t^i(x_t^j)
    for t=1:T
        switch performance_metric
            case 'Individual_Regret' % INDIVIDUAL REGRET------------------------------------------
                % Setting the performance metric -------------------------------
                % Each private function FI is evaluated in X_t^j, so X_t^j is
                % repeated n times (number of agents).
                [~,F_saved_fixed_agent(:,t)] = LocalOracles(FI(:,t),repmat(X(fixed_agent_j,t),1,n));
            case 'Averaged_Individual_Regret' % NO INDIVIDUAL AVERAGE REGRET---------------------------------
                % Here it's the same thing than in case 1 bu with Xaveraged
                % instead of X
                [~,F_saved_fixed_agent(:,t)] = LocalOracles(FI(:,t),repmat(Xaveraged(fixed_agent_j,t),1,n));
            otherwise
                error("Unknown performance metric");
        end
        F_sum_fixed = F_sum_fixed + sumcell(F_saved_fixed_agent(:,t)); % The regret is augmented
    end
    P.PerformanceMetric(F_sum_fixed - fs);
    % Activate the trace heuristic for trying to reduce the solution dimension
    % P.TraceHeuristic(1); % uncomment to activate
    
    % (6) Solving the PEP -------------------------------------------------
    if verbose
        switch type
            case 'spectral_relaxed'
                fprintf("Spectral PEP formulation for DGD after %d iterations, with %d agents \n",T,n);
                fprintf("Using the following spectral range for the averaging matrix: [%1.2f, %1.2f] \n",mat(1),mat(2))
            case 'exact'
                fprintf("Exact PEP formulation for DGD after %d iterations, with %d agents \n",T,n);
                fprintf("The used averaging matrix is\n")
                disp(mat);
        end
    end

    % Values to be used in the upper bound
    [gw_u,w_u,~] = w.oracle(u); 
    sum_Bregman_first_estimates_PEP = n*w_u - n*(gw_u*u);
    for i=1:n
        [~,w_xi1,~] = w.oracle(X{i,1});
        sum_Bregman_first_estimates_PEP = sum_Bregman_first_estimates_PEP - w_xi1 + gw_u*X{i,1};
    end

    % DEBUGGING: ----------------------------------------------------
    %offline_local_cost = cell(n,T);
    for t=1:T
        [~, offline_local_cost(:,t)] = LocalOracles(FI(:,t),repmat({u},1,n));
    end
    % ---------------------------------------------------------------

    % Adding constraints to add numericall stability
    for t=1:T
        for i=1:n
            P.AddConstraint((X{i,t}-u)^2<=D^2);
        end       
    end
    

    out = P.solve(verbose);

    if verbose, out, end


    % Evaluate the output ---------------------------------------------
    wc = out.WCperformance;
    
    % DEBUGGING: Checking the local regret ---------------------------
    % fprintf("Offline total cost=%d\n",double(fs));

    if verbose
        for t=1:T
            fprintf("---------------t=%d-------------------\n",t)
            for i=1:n
                fprintf("f[%d,%d](x_jt)=%d \n ",i,t,double(F_saved_fixed_agent{i,t}));
                fprintf("g[%d,%d](x_jt)=%d \n ",i,t,norm(double(G_saved{i,t})));
                fprintf("f[%d,%d](u)=%d \n ",i,t,double(offline_local_cost{i,t}));
            end
        end
    
        fprintf("==============Checking estimates=============\n")
        for t=1:T
            for i=1:n
                fprintf("X{%d,%d} - u=%d \n",i,t,norm(double(X{i,t}-u)))
                fprintf("Y{%d,%d} - u=%d \n",i,t,norm(double(Y{i,t}-u)))
                for s=1:T
                    for j=1:n
                        fprintf("X{%d,%d} - X{%d,%d}=%d \n",i,t,j,s,norm(double(X{i,t}-X{j,s})))
                        fprintf("Y{%d,%d} - X{%d,%d}=%d \n",i,t,j,s,norm(double(Y{i,t}-X{j,s})))
                    end
                end
            end
        end
    end
    % ------------------------------------------------------------
    
    % Construct an approximation of the worst averaging matrix that links the solutions X and Y
    [Ah.X,Ah.r,Ah.status] = A.estimate(0);
    % where Ah.X is the worst-case communication matrix
    %display(Ah.X)
    %display(Ah.status)
    % Computes the smallest non null-entries of the estimated worst-case
    % matrix
    min_val = smallestNonNull(Ah.X,THRESHOLD_MIN_VAL_DOMD) ;

    % sum_norm_first_iterates
    sum_norm_first_estimates = 0;
    for i=1:n
        sum_norm_first_estimates = sum_norm_first_estimates + norm(double(X{i,1}));
    end
    
    sum_Bregman_first_estimates = double(sum_Bregman_first_estimates_PEP);
end