% Function which returns the TIGHTEST bound for the Distributed Online
% Conditional Gradient introduced by Zhang, W., Zhao, P., Zhu, W., Hoi,
% S. C., & Zhang, T. (2017, July). Projection-free distributed online 
% learning in networks. In International conference on machine learning 
% (pp. 4054-4062). PMLR.

% Input:
% T: Number of iterations
% D: Diameter of the feasible set
% L: Lipschitz Constant i.e., bound over the magnitude of gradients
% n: number of agents
% performance_metric: 1 --> Classic Individual Regret; 2 --> Averaged Individual Regret
% verbose: 0 --> No verbose, 1 --> Full verbose
% sigma: Bound for the second-largest eigenvalue of the communication
% network
function [wc]=distributed_online_conditional_gradient(T,D,L,n,performance_metric,verbose,sigma)
   % Setting the network topology ----------------------------------------
    type = 'spectral_relaxed'; % The topology is defined by the spectrum of Consensus matrix 
    time_varying_mat = 0; % The communication is supposed static
    returnOpt = 0;
    equalStart = 0; % Estimates are different from each other
    mat = [-sigma,sigma];

    % Set up general problem parameters -----------------------------------
    % In the paper step-sizes are indexes wrt the number of agents 'eta_i'
    % instead of the current time-step
    compute_step_size = @(i) (1-sigma)*D / (2*(sqrt(n)+1+(sqrt(n)-1)*sigma)*L*T^(3/4));
    sigmait = @(i,t) 1/sqrt(t); 
    
    % Initialize an empty PEP
    P = pep();
    
    % Defining an indicator function
    param_id.D = D;
    id_operator = P.DeclareFunction('ConvexIndicator',param_id);
    
    % Set up the local and global objective functions -----------------
    fctClass = 'ConvexBoundedGradient'; % Class of functions to consider for the worst-case
    fctParam.R = L; % Bounded Gradient constant ||g||^2 <= L^2.
    
    FI = cell(n, T+1); % A matrix of mxT functions

    for t=1:T+1
        [Fi,~,~,~] = P.DeclareMultiFunctions(fctClass,fctParam,n,returnOpt);
        for i=1:n
            FI(i,t) = {Fi{i}}; % Before it was FI(i,t) = Fi(i) + id_operator but it makes no sense
        end
    end    

    % Iterates cells 
    % Storing estimates
    Z = cell(n, T+1);        
    X = cell(n, T+1);           
    V = cell(n, T); 
    DIR = cell(n,T);
    
    % F_saved = cell(n,T+1); ---> Useless since we use the individual
    % regret as main performance metric
    G_saved = cell(n,T+1);
    F_saved = cell(n,T+1);

    % Set up the starting points and initial conditions -------------------
    X_init = P.MultiStartingPoints(n,equalStart);
    for i=1:n
        X(i,1) = {projection_step(X_init{i},id_operator)};
    end
    [G_saved(:,1),F_saved(:,1)] = LocalOracles(FI(:,1),X(:,1));
    
    % Set up the averaging matrix -----------------------------------------
    A = P.DeclareConsensusMatrix(type,mat,time_varying_mat);

    % Constraining all estimates Z(i,1) to be zero for all i 
    Z(:,1)=P.MultiStartingPoints(n,1); % equal start for Z which are all equal to zero
    for i=1:n
        P.AddConstraint(Z{i,1}^2==0);
    end
    
    Xaveraged = X; 
    % Ensuring that first estimates are in the feasible set
    % Algorithm (Distributed Online Dual Averaging) -----------------------
    % Iterations     
    for t = 1:T
        for i=1:n
            % Computing the step-size and the mixing coefficient sigmait
            eta = compute_step_size(i); % in the paper eta depends on the number of agents
            mix_val = sigmait(i,t);
            if verbose
                fprintf("i=%d \n",i);
                fprintf("t=%d \n",t);
                fprintf("eta=%f \n",eta);
                fprintf("mix_val=%f \n",mix_val);
            end
            DIR(i,t) = {eta *  Z{i,t} + 2*(X{i,t}-X{1,1})}; % Obtained from computing naba F_t,i from the paper 
            V(i,t) = {linearoptimization_step(DIR{i,t},id_operator)}; % Find the minimizer following dir while ensuring that the results is in the feasible set
            X(i,t+1) = {X{i,t} + mix_val * (V{i,t}-X{i,t})};  
            % Averaged estimate (COMPUTED BUT NOT CONSIDERED BY DEFAULT) 
            Xaveraged(i,t+1) = {sumcell(X(i,1:t+1))/(t+1)};
        end
        AxZ = A.consensus(Z(:,t)); % Averaging step
        for i=1:n
            Z(i,t+1) = {AxZ{i} + G_saved{i,t}}; % Final update
        end
        [G_saved(:,t+1),F_saved(:,t+1)] = LocalOracles(FI(:,t+1),X(:,t+1)); 
    end
    
    % Defining a comparison point which is not necessarily an optimal point
    % in the paper
    unconstrained_u = Point('Point'); 
    u = projection_step(unconstrained_u,id_operator);
    % Without loss of generality
    P.AddConstraint(u^2 ==0);

    % SETTING THE PERFORMANCE METRIC --------------------------------------
    fixed_agent_j = 1; % The regret is the same regardless the agent.
    F_sum_fixed = 0;
    F_saved_fixed_agent = cell(n,T); % Each cell contains F_i,t(x_t^j)
    fs_t = cell(n,T); % Each cell contains F_i,t(u)
    fs = 0; % sum_i,t f_it(u)
    for t=1:T
        switch performance_metric
            case 'Individual_Regret' % INDIVIDUAL REGRET------------------------------------------
                % Setting the performance metric -------------------------------
                % Each private function FI is evaluated in X_t^j, so X_t^j is
                % repeated n times (number of agents).
                [~,F_saved_fixed_agent(:,t)] = LocalOracles(FI(:,t),num2cell(repmat(X{fixed_agent_j,t},1,n)));
            case 'Averaged_Individual_Regret' % NO INDIVIDUAL AVERAGED REGRET---------------------------------
                % Here it's the same thing than in case 1 bu with Xaveraged
                % instead of X
                [~,F_saved_fixed_agent(:,t)] = LocalOracles(FI(:,t),num2cell(repmat(Xaveraged{fixed_agent_j,t},1,n)));
            otherwise 
                error("Unknown Perfomance Metric");
        end
        % The total cost is augmented
        F_sum_fixed = F_sum_fixed + sumcell(F_saved_fixed_agent(:,t)); 
        % The cost in u is cummulated too
        [~, fs_t(:,t)] = LocalOracles(FI(:,t),num2cell(repmat(u,1,n)));  
        fs = fs + sumcell(fs_t(:,t));
    end
    P.PerformanceMetric(F_sum_fixed - fs);

   % Solving the PEP -------------------------------------------------
   if verbose
        switch type
            case 'spectral_relaxed'
                fprintf("Spectral PEP formulation for DOCG after %d iterations, with %d agents \n",T,n);
                fprintf("Using the following spectral range for the averaging matrix: [%1.2f, %1.2f] \n",mat)
            case 'exact'
                fprintf("Exact PEP formulation for DOCG after %d iterations, with %d agents \n",T,n);
                fprintf("The used averaging matrix is\n")
                disp(mat);
        end
   end

   traceHeuristicActivated = 0;
   P.TraceHeuristic(traceHeuristicActivated);
   out = P.solve(verbose);

   if verbose, out, end

    % Evaluate the output ---------------------------------------------
    wc = out.WCperformance;
end