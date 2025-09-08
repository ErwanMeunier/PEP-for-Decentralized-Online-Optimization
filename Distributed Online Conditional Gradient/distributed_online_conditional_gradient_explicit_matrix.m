% Function which returns the TIGHTEST bound for the Distributed Online
% Conditional Gradient

% Input: TODO

% performance_metric : 1 --> Classic Individual Regret; 2 --> Average
% Individual Regret
function [X_output,F_output,G_output,Z_output,V_output,Fu_output,Gu_output,compute_step_size,sigmait,wc]=distributed_online_conditional_gradient_explicit_matrix(T,D,L,performance_metric,verbose,mat)
    % Setting the network topology ----------------------------------------
    n = size(mat,1);
    type = 'exact'; % The topology is defined by the spectrum of Consensus matrix 
    time_varying_mat = 0; % The communication is supposed static
    returnOpt = 0;
    equalStart = 0; % Estimates are different from each other
    if n>1
        evs = sort(eig(mat),'descend'); % Computing and sorting in descending order the eigenvalues of the communication matrix 
        sigma = evs(2); % sigma is the second largest eigenvalues 
    else
        sigma = 1;
    end

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
    
    % Constraining all first estimates to be feasible 
    % TO DO in the future: Using the initial conditons operator
    
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
            % FROM THE PAPER: DIR(i,t) = {eta *  Z{i,t} + 2*(X{i,t}-X{1,1})}; % Obtained from computing naba F_t,i from the paper 
            % Debugging:
            DIR(i,t) = {eta*Z{i,t} + 2*(X{i,t}-X{1,1})};
            %%%%%%%%
            % FROM THE PAPER: V(i,t) = {linearoptimization_step(DIR{i,t},id_operator)}; % Find the minimizer following dir while ensuring that the results is in the feasible set
            % Debugging
            V(i,t) = {linearoptimization_step(DIR{i,t},id_operator)};
            %%%%%%%%
            % Descent step
            % FROM THE PAPER: X(i,t+1) = X{i,t} + mix_val * (V{i,t}-X{i,t})}; 
            % Debugging:
            X(i,t+1) = {X{i,t} + mix_val * (V{i,t}-X{i,t})};  
            %%%%%%%%
            % Averaged estimate (COMPUTED BUT NOT CONSIDERED) 
            Xaveraged(i,t+1) = {sumcell(X(i,1:t+1))/(t+1)};
        end
        AxZ = A.consensus(Z(:,t)); % Averaging step
        for i=1:n
            % FROM THE PAPER : Z(i,t+1) = {AxZ{i} + G_saved{i,t}}; % Final update
            % Debugging:
            Z(i,t+1) = {AxZ{i}+G_saved{i,t}};
            %%%%%%%%
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
    G_saved_fixed_agent = cell(n,T);
    fs_t = cell(n,T); % Each cell contains F_i,t(u)
    gs_t = cell(n,T); % Each cell contains subgradient F_{i,t}(u) 
    fs = 0; % sum_i,t f_it(u)
    for t=1:T
        switch performance_metric
            case 'Individual_Regret' % INDIVIDUAL REGRET------------------------------------------
                % Setting the performance metric -------------------------------
                % Each private function FI is evaluated in X_t^j, so X_t^j is
                % repeated n times (number of agents).
                [G_saved_fixed_agent(:,t),F_saved_fixed_agent(:,t)] = LocalOracles(FI(:,t),num2cell(repmat(X{fixed_agent_j,t},1,n)));
            case 'Averaged_Individual_Regret' % NO INDIVIDUAL AVERAGE REGRET---------------------------------
                % Here it's the same thing than in case 1 bu with Xaveraged
                % instead of X
                [G_saved_fixed_agent(:,t),F_saved_fixed_agent(:,t)] = LocalOracles(FI(:,t),num2cell(repmat(Xaveraged{fixed_agent_j,t},1,n)));
        end
        % The total cost is augmented
        F_sum_fixed = F_sum_fixed + sumcell(F_saved_fixed_agent(:,t)); 
        % The cost in u is cummulated too
        [gs_t(:,t), fs_t(:,t)] = LocalOracles(FI(:,t),num2cell(repmat(u,1,n)));  
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
   out = P.solve(1);

   %if verbose, out, end

    % Evaluate the output ---------------------------------------------
    wc = out.WCperformance;

    % Construct an approximation of the worst averaging matrix that links the solutions X and Y
    [Ah.X,Ah.r,Ah.status] = A.estimate(0);
    if verbose 
        % Recovering the worst-case averaging matrix
        fprintf('Estimation of the worst-case averging matrix:\n');
        disp(Ah.X);
        fprintf("Residual norm of the estimated averaging matrix: %d \n",Ah.r);
        disp(Ah.status);
        % Recovering the estimates
        fprintf("Computed estimates are:\n");
        normXIT = cell(n,T+1);
        for t=1:T
            for i=1:n
                normXIT(i,t) = {norm(double(X{i,t}))};
            end
        end
        fprintf("norm(X(i,t))=\n");
        disp(cell2mat(normXIT)),
        if traceHeuristicActivated
            for t=1:T
                for i=1:n
                    fprintf("X{%d,%d}=\n",i,t);
                    disp(double(X{i,t}));
                end
            end
        end
        %%%%%%%%%%%%%%%%%%
        normVIT = cell(n,T);
        for t=1:T
            for i=1:n
                 normVIT(i,t) = {norm(double(V{i,t}))};
            end
        end
        fprintf("norm(V(i,t))=\n");
        disp(cell2mat(normVIT));
        %%%%%%%%%%%%%%%%%%
        normDirIT = cell(n,T);
        for t=1:T
            for i=1:n
                 normDirIT(i,t) = {norm(double(DIR{i,t}))};
            end
        end
        fprintf("norm(Dir(i,t))=\n");
        disp(cell2mat(normDirIT));
        %%%%%%%%%%%%%%%%%
        normZIT = cell(n,T);
        for t=1:T
            for i=1:n
                 normZIT(i,t) = {norm(double(Z{i,t}))};
            end
        end
        fprintf("norm(Z(i,t))=\n");
        disp(cell2mat(normZIT));
        %%%%%%%%%%%%%%%%%%
        normGIT = cell(n,T+1);
        for t=1:T
            for i=1:n
                 normGIT(i,t) = {norm(double(G_saved{i,t}))};
            end
        end
        fprintf("norm(G(i,t))=\n");
        disp(cell2mat(normGIT));
        %%%%%%%%%%%%%%%%%%
        normGIT_fixed_agent = cell(n,T+1);
        for t=1:T
            for i=1:n
                 normGIT_fixed_agent(i,t) = {norm(double(G_saved_fixed_agent{i,t}))};
            end
        end
        fprintf("norm(G_i,t(x_j,t))=\n");
        disp(cell2mat(normGIT_fixed_agent));
        %%%%%%%%%%%%%%%%%%
        fprintf("Local costs \n");
        % Displaying incurred costs for each agent in its estimate
        FIT = cell(n,T);
        for i=1:n
            for t=1:T
                FIT(i,t) = {double(F_saved{i,t})};
            end
        end
        fprintf("Costs for each agent in its estimate:\n --> f_it(x_it)=\n");
        % Displaying
        disp(cell2mat(FIT));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        FIT_indj = cell(n,T);
        for i=1:n
            for t=1:T
                FIT_indj(i,t) = {double(F_saved_fixed_agent{i,t})};
            end
        end
        fprintf("Costs for each agent in its agent j's estimates:\n --> f_it(x_jt)=\n");
        disp(cell2mat(FIT_indj));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        FS_u = cell(n,T);
        fprintf("Costs for each agent in its agent u:\n --> f_it(u)=\n");
        for i=1:n
            for t=1:T
                FS_u(i,t) = {double(fs_t{i,t})};
            end
        end
        disp(cell2mat(FS_u));
        % Verifying that distances between estimates are bounded by D^2
        ok_for_the_diameter = true;
        for t=1:T
            for tau=1:T
                for i=1:n
                    for j=1:n
                        %fprintf("X{%d,%d}-X{%d,%d}=%d <= %d \n ",i,t,j,tau,norm(double(X{i,t}-X{j,tau})),D);
                        %fprintf("Checking diameter: %d \n",(norm(double(X{i,t}-X{j,tau})) <= D+0.001)); 
                        ok_for_the_diameter = ok_for_the_diameter && (norm(double(X{i,t}-X{j,tau})) <= D);
                    end
                end
            end
        end
    fprintf("Ok for diameter ? %d \n",ok_for_the_diameter);
    fprintf("The second largest eigenvalue is: sigma=%f",sigma);
    out % Displaying the output of PEP
    end

    % OUTPUT 
    X_output = cellfun(@double,X,'UniformOutput',false);
    F_output = cellfun(@double,F_saved,'UniformOutput',false);
    G_output = cellfun(@double,G_saved,'UniformOutput',false);
    Z_output = cellfun(@double,Z,'UniformOutput',false);
    V_output = cellfun(@double,V,'UniformOutput',false);
    Fu_output= cellfun(@double,fs_t,'UniformOutput',false);
    Gu_output= cellfun(@double,gs_t,'UniformOutput',false);
end