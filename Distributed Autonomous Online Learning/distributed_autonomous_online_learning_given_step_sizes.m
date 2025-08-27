% Function which returns the TIGHTEST bound for the Distributed Autonomous Online Learning algorithm 
% (DAOL) depicted in F. Yan, S. Sundaram, S. V. N. Vishwanathan and Y. Qi, 
% "Distributed Autonomous Online Learning: Regrets and Intrinsic Privacy-Preserving Properties," 
% in IEEE Transactions on Knowledge and Data Engineering, vol. 25, no. 11, pp. 2483-2493, Nov. 2013, 
% doi: 10.1109/TKDE.2012.191.

% T: Number of iterations of the gradient method
% m: Number of agents
% F: Diameter over the compact set
% mu: Strong convexity constant for the objective function
% L: Lipschitz coefficient
% mat_vec: Communication Matrix (if fixed matrix) / Matrix range (if spectral).
% mat_vec is given under vector format.
% relaxed)

% Example for calling the function: distributed_autonomous_online_learning(20,2,0.5,1,1,'spectral_relaxed',[0,1],0,1)

function [wc]=distributed_autonomous_online_learning_given_step_sizes(T,D,L,mu,n,performance_metric,verbose,sigma,compute_step_size)
   % Setting the network topology ----------------------------------------
    type = 'spectral_relaxed'; % The topology is defined by the spectrum of Consensus matrix 
    time_varying_mat = 0; % The communication is supposed static
    mat = [-sigma,sigma]; % range of eigenvalues for the communication matrix
    returnOpt = 0; 
    equalStart = 1; % Assumption 6 of the paper --> X(i,1) are identically initialized 

    % Setting-up step-sizes -----------------------------------
    %%%%%%%%%%%%%%%% GIVEN %%%%%%%%%%%%%%%%%

    % Initialize an empty PEP
    P = pep();
    
    % (1) Set up the local and global objective functions -----------------
    fctClass = 'StronglyConvexBoundedGradient'; % Class of functions to consider for the worst-case
    fctParam.mu = mu; % mu strongly convex function
    fctParam.R = L; % Bounded Gradient constant ||g||^2 <= G^2.
    
    FI = cell(n, T+1); % A matrix of mxT functions

    for t=1:T+1
        [Fi,~,~,~] = P.DeclareMultiFunctions(fctClass,fctParam,n,returnOpt);
        FI(:,t) = Fi;
    end

    % BOUNDED DOMAIN VIA INDICATOR FUNCTION
    param_id.D = D; % Diameter of the ball defining the compact set: ||x-y||^2 <= F^2, for all x and y
    project_function = P.DeclareFunction('ConvexIndicator',param_id);
    % Getting the optimal point and value to compute the regret in due time
    F_sum_with_id = project_function; % J(w) in paper.
    for t=1:T
        F_sum_with_id = F_sum_with_id + sumcell(FI(:,t));
    end
    % Defining the optimal point to consider the regret as the performance metric 
    [u, fs] = F_sum_with_id.OptimalPoint();
    
    % Iterates cells 
    X_hat = cell(n, T);         % X_hat = A * X - eta_t*gradient(F(X))
    X = cell(n, T+1);           % FEASIBLE: local estimates = Proj(Z,id(H))
   
    %F_saved = cell(m,T+1); NO NEED TO SAVE IT
    G_saved = cell(n,T+1);

    % Set up the starting points and initial conditions -------------------
    X(:,1) = P.MultiStartingPoints(n,equalStart);
    [G_saved(:,1),~] = LocalOracles(FI(:,1),X(:,1)); % F_1(X(1,1)), F_2(X(2,1)), ...
    P.AddMultiConstraints(@(xi) (xi-u)^2 <= D^2, X(:,1)); %initial condition: ||wi0 - ws||^2 <= F^2

    % Set up the averaging matrix -----------------------------------------
    A = P.DeclareConsensusMatrix(type,mat,time_varying_mat);

    % Algorithm (DGD) -----------------------------------------------------
    % Iterations 
    X_hat(:,1)=X(:,1);
    for t = 1:T
        eta = compute_step_size(t);
        % Mixing and gradient
        AxX = A.consensus(X(:,t));
        % Projection 
        %disp(AxX)
        for i=1:n
            X_hat(i,t+1) = {AxX{i} - eta*G_saved{i,t}};
            X(i,t+1) = {projection_step(X_hat{i,t+1},project_function)};
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

    out = P.solve(verbose);

    if verbose, out, end

    % (7) Evaluate the output ---------------------------------------------
    wc = out.WCperformance;
    %double(X{1,1})
    % (8) Construct an approximation of the worst averaging matrix that links the solutions X and Y
    [Ah.X,Ah.r,Ah.status] = A.estimate(0);
end