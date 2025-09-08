% PATH AND FILESYSTEM UTILITIES
OUTPUT_DIR = "./Simulation_Results";

% Defaults parameters
T_def=10; % Number of iterations
N_def=2; % Number of agents 
D_def=1; % Diameter for the feasible set
L_def=1; % Lipschitz constant for the private objective functions
G_def=2; % Smoothness constant for the Kernel function
mu_def=0; % Strong convexity constant for private objective functions
sigma_def=0.9; % Second largest eigenvalue
epsilon_def=(1-sigma_def)/2; % sigma2 = 1 - 2 epsilon <=> epsilon = (1-sigma2)/2 
lambda_def=1; % Strong convexity constant for the Kernel
verbose_def = 0;
performance_metric_def = 'Individual_Regret';

% Threshold for the min value extracted for computing the bound of DOMD
THRESHOLD_MIN_VAL_DOMD = 10^(-6);

% Benchmarking parameters
NB_SAMPLES_SIGMA = 20;
LB_T = 1; % Lower bound over the number of iterations  
UB_T = 20;% Upper bound over the number of iterations 
LB_N = 1; % Lower bound over the number of agents
UB_N = 5; % Upper bound over the number of agents

% Choice over which simulations that should be carried on
perform_allsimulations = true;
sim1and3 = false || perform_allsimulations;
sim2 = false || perform_allsimulations;
sim4 = false || perform_allsimulations;

% Cells to save the results
if sim1and3
    Results1and3 = cell(3,3,NB_SAMPLES_SIGMA); % Number of methods x PEP or Literature Bound  or Parameter value x NB_SAMPLES_SIGMA
end
if sim2
    Results2 = cell(3,3,UB_T-LB_T+1); % Number of methods x PEP or Literature Bound or Parameter value x (UB_T - LB_T + 1)
end
if sim4
    Results4 = cell(3,3,UB_N-LB_N+1); % Number of methods x PEP or Literature Bound or Parameter value x (UB_N - LB_N + 1)
end
% Convention: 1: DAOL, 2: DOCG, 3: DOMD
% Convention: 1: PEP bound, 2: Literature bound, 3: Parameter value

if sim1and3
	sigma_values = linspace(0.01, 0.99, NB_SAMPLES_SIGMA);
    % Figure 1: Normalized Regret v.s. Bounds coming from the literature w.r.t. Second Largest Eigenvalue
    fprintf("Computing data for Figures 1 and 3--> DAOL + DOCG... \n");
    for i=1:NB_SAMPLES_SIGMA
        fprintf("SIMULATION_1and3 DAOL+DOCG: Progress: %d %% \n", i/NB_SAMPLES_SIGMA * 100);
        sigma = sigma_values(i);
        % PEP bounds
        Results1and3{1,1,i} =  distributed_autonomous_online_learning(T_def,D_def,L_def,mu_def,N_def,performance_metric_def,verbose_def,sigma);
        Results1and3{2,1,i} =  distributed_online_conditional_gradient(T_def,D_def,L_def,N_def,performance_metric_def,verbose_def,sigma);
        % Literature bounds
        Results1and3{1,2,i} =  bound_daol(T_def,D_def,L_def,mu_def,N_def,performance_metric_def,verbose_def,sigma);
        Results1and3{2,2,i} =  bound_docg(T_def,D_def,L_def,N_def,performance_metric_def,verbose_def,sigma); % ok
        % Value for the parameters
        Results1and3{1,3,i} = sigma;
        Results1and3{2,3,i} = sigma;
    end

    % DOMD is simulated w.r.t. epsilon (lower bound on the entries) and not the second largest eigenvalue
    % THIS MUST BE USED ONLY IF N_DEF=2
    epsilon_values = linspace(0,0.5,NB_SAMPLES_SIGMA);
	fprintf("Computing data for Figures 1 and 3--> DOMD... \n");
    for i=1:NB_SAMPLES_SIGMA
        fprintf("SIMULATION_1and3 DOMD: Progress: %d %% \n", i/NB_SAMPLES_SIGMA * 100);
        epsilon = epsilon_values(i);
        sigma_bound = 1-2*epsilon;
        [wc,sum_norm_first_estimates,sum_Bregman_first_estimates,eta,min_val] =  distributed_mirror_descent_online_optimization(T_def,D_def,L_def,G_def,lambda_def,N_def,performance_metric_def,verbose_def,sigma_bound,THRESHOLD_MIN_VAL_DOMD);
        Results1and3{3,1,i} = wc;
        Results1and3{3,2,i} = bound_DOMD(T_def,D_def,L_def,G_def,lambda_def,N_def,performance_metric_def,verbose_def,sum_norm_first_estimates,sum_Bregman_first_estimates,eta,epsilon);
        Results1and3{3,3,i} = epsilon;
    end 
end
% Figure 2: Normalized Regret w.r.t. Number of iterations
if sim2
    fprintf("Computing data for Figure 2 --> DAOL + DOCG... \n");
    for i=1:(UB_T-LB_T+1)
        fprintf("SIMULATION_2: Progress: %d %% \n", i/(UB_T-LB_T+1) * 100);
        T=LB_T+i-1;
        % PEP bounds
        Results2{1,1,i} =  distributed_autonomous_online_learning(T,D_def,L_def,mu_def,N_def,performance_metric_def,verbose_def,sigma_def);
        Results2{2,1,i} =  distributed_online_conditional_gradient(T,D_def,L_def,N_def,performance_metric_def,verbose_def,sigma_def);
        % Literature bounds
        Results2{1,2,i} =  bound_daol(T,D_def,L_def,mu_def,N_def,performance_metric_def,verbose_def,sigma_def);
        Results2{2,2,i} =  bound_docg(T,D_def,L_def,N_def,performance_metric_def,verbose_def,sigma_def); % ok
        % Value for the parameter
        Results2{1,3,i} = T;  
        Results2{2,3,i} = T;

        sigma_bound = 1-2*epsilon_def;
        [wc,sum_norm_first_estimates,sum_Bregman_first_estimates,eta,min_val] =  distributed_mirror_descent_online_optimization(T,D_def,L_def,G_def,lambda_def,N_def,performance_metric_def,verbose_def,sigma_bound,THRESHOLD_MIN_VAL_DOMD);
        Results2{3,1,i} = wc;
        Results2{3,2,i} =  bound_DOMD(T,D_def,L_def,G_def,lambda_def,N_def,performance_metric_def,verbose_def,sum_norm_first_estimates,sum_Bregman_first_estimates,eta,epsilon_def);
        Results2{3,3,i} = T;
    end
end
% Figure 3: Normalized Regret w.r.t. Second Largest Eigenvalue (comparing --> which method is the best one)
% --> Results could be obtained with those from figure 1 --> Thats why it is Results1and3
% NOTHING HERE
% Figure 4: Normalized Regret w.r.t. Number of Agents
if sim4
    fprintf("SIMULATION_4: Computing data for Figure 4 --> DAOL + DOCG + DOMD\n");
    for i = 1:(UB_N-LB_N+1)
        fprintf("Progress: %d %% \n", i/(UB_N-LB_N+1) * 100);
        N=LB_N+i-1;
        % PEP bounds
        Results4{1,1,i} =  distributed_autonomous_online_learning(T_def,D_def,L_def,mu_def,N,performance_metric_def,verbose_def,sigma_def);
        Results4{2,1,i} =  distributed_online_conditional_gradient(T_def,D_def,L_def,N,performance_metric_def,verbose_def,sigma_def);
        % Literature bounds
        Results4{1,2,i} =  bound_daol(T_def,D_def,L_def,mu_def,N,performance_metric_def,verbose_def,sigma_def);
        Results4{2,2,i} =  bound_docg(T_def,D_def,L_def,N,performance_metric_def,verbose_def,sigma_def); % ok
        % Value for the parameter
        Results4{1,3,i} = N;
        Results4{2,3,i} = N;

        %sigma_bound = 1-2*epsilon_def;
        %[wc,sum_norm_first_estimates,sum_Bregman_first_estimates,eta] =  distributed_mirror_descent_online_optimization(T_def,D_def,L_def,G_def,lambda_def,N_def,performance_metric_def,verbose_def,sigma_bound);
        [wc,sum_norm_first_estimates,sum_Bregman_first_estimates,eta,min_val] =  distributed_mirror_descent_online_optimization(T_def,D_def,L_def,G_def,lambda_def,N,performance_metric_def,verbose_def,sigma_def,THRESHOLD_MIN_VAL_DOMD);
        Results4{3,1,i} = wc;
        fprintf("Estimated worst-case epsilon: %d \n ", min_val); % the espilon is computed according to the non null min-value of the network matrix
        Results4{3,2,i} = bound_DOMD(T_def,D_def,L_def,G_def,lambda_def,N,performance_metric_def,verbose_def,sum_norm_first_estimates,sum_Bregman_first_estimates,eta,min_val);
        Results4{3,3,i} = N;
    end
end

%%%%%%%%%%%%%%%%%% EXPORTING RESULTS %%%%%%%%%%%%%%%%%%%

path_results1and3 = fullfile(OUTPUT_DIR,...
                        sprintf("results1and3_T%dN%dD%dL%dG%dmu%dlambda%d.csv",...
                                    T_def,N_def,D_def,L_def,G_def,mu_def,lambda_def)...
                    ); % 
path_results2 = fullfile(OUTPUT_DIR,...
                        sprintf("results2_N%dD%dL%dG%dmu%dsigma%depsilon%dlambda%d.csv",...
                                    N_def,D_def,L_def,G_def,mu_def,sigma_def,epsilon_def,lambda_def)...
                    ); % T is the variable
path_results4 = fullfile(OUTPUT_DIR,...
                        sprintf("results4_T%dD%dL%dG%dmu%dsigma%depsilon%dlambda%d.csv",...
                                    T_def,D_def,L_def,G_def,mu_def,sigma_def,epsilon_def,lambda_def)...
                    ); % N is the variable

% Creating / Overwritting files if they already exist
filenames = [path_results1and3,path_results2,path_results4];

emergency_saving = false;  % BE VERY CAREFUL WITH THIS PARAMETER. To be used 
% only if simulations have been performed and that something went wrong
% with the export of data.

for i=1:3 % avoid overwriting file if no simulation has been performed 
    writing = true;
    switch i 
        case 1
            writing = sim1and3;
        case 2 
            writing = sim2;
        case 3
            writing = sim4;
    end
    filename = filenames(i);
    if writing || emergency_saving
        fprintf("Creating %s \n",filename);
        fid = fopen(filename,'w');
        fprintf(fid,'');         
        fid = fclose(fid);
    end
end

% Saving results into files
% avoid overwriting file if no simulation has been performed 
if sim1and3
    fprintf("Writing in file %s \n",path_results1and3);
    writecell(Results1and3,path_results1and3,'FileType','text');
end
if sim2
    fprintf("Writing in file %s \n",path_results2);
    writecell(Results2,path_results2,'FileType','text');
end
if sim4
    fprintf("Writing in file %s \n",path_results4);
    writecell(Results4,path_results4,'FileType','text');
end

%%%%%%%%%%% POST-PROCESSING and PLOTTING %%%%%%%%%%%%%
% Here our code is meant to be executed appart from the previous code
% Which means that we extract the results from the results files.
% We only use filenames: path_results1and3, path_results2, path_results4

% Reading files (data have been flattened) 
r1and3_2D = readcell(path_results1and3);
r2_2D = readcell(path_results2);
r4_2D = readcell(path_results4);

% Flattening
r1and3_linear = r1and3_2D(:); 
r2_linear = r2_2D(:);
r4_linear = r4_2D(:);

% Reshaping arrays 
r1and3 = reshape(r1and3_linear,[3,3,NB_SAMPLES_SIGMA]);
r2 = reshape(r2_linear,[3,3,UB_T-LB_T+1]);
r4 = reshape(r4_linear,[3,3,UB_N-LB_N+1]);

%%%%%%%%%%%%%%%%%% CRAFTING THE FIGURES %%%%%%%%%%%%%%%%%%%
% Normalizing the Regret: Regret_normalized = Regret / TNLD

% Figure 1 and 3: Normalized Regret v.s. Bounds coming from the literature w.r.t. Second Largest Eigenvalue
% PEP bounds
regret_DAOL_wrtSigma = cell2mat({r1and3{1,1,:}}) / (T_def * D_def * L_def * N_def);
regret_DOCG_wrtSigma = cell2mat({r1and3{2,1,:}}) / (T_def * D_def * L_def * N_def);
regret_DOMD_wrtEpsilon = cell2mat({r1and3{3,1,:}}) / (T_def * D_def * L_def * N_def);

% Bounds from the literature
regret_LitDAOL_wrtSigma = cell2mat({r1and3{1,2,:}}) / (T_def * D_def * L_def * N_def);
regret_LitDOCG_wrtSigma = cell2mat({r1and3{2,2,:}}) / (T_def * D_def * L_def * N_def);
regret_LitDOMD_wrtEpsilon = cell2mat({r1and3{3,2,:}}) / (T_def * D_def * L_def * N_def);

% x-axis values
range_sigma = cell2mat({r1and3{1,3,:}}); % Same range for DAOL and DOCG
range_epsilon = cell2mat({r1and3{3,3,:}}); % Dedicated range for DOMD USELESS FOR NOW SINCE WE DO HAVE THE MAPPING BETWEEN sigma and epsilon

fig1and3 = figure;


% Literature bounds
subplot(2,1,1);
hold on; 
title('Literature Bounds (log-scale)')
% regret_LitDOMD_wrtEpsilon and regret_DOMD_wrtEpsilon are flipped due to the mapping between sigma and epsilon
semilogy(range_sigma,regret_LitDAOL_wrtSigma,'r--', 'LineWidth', 1.5, 'DisplayName', 'DAOL', 'MarkerFaceColor', 'r');
semilogy(range_sigma,regret_LitDOCG_wrtSigma, 'b--', 'LineWidth', 1.5, 'DisplayName', 'DOCG', 'MarkerFaceColor', 'b');
semilogy(range_sigma,flip(regret_LitDOMD_wrtEpsilon), 'g--', 'LineWidth', 1.5, 'DisplayName', 'DOMD', 'MarkerFaceColor', 'g');
set(gca, 'YScale', 'log');
xlabel('$\lambda_2$','Interpreter','latex');
ylabel('$\overline{\mathbf{Reg}}_j(T)$','Interpreter','latex');
grid on;
hold off
% PEP bounds 
subplot(2,1,2);
hold on
title('PEP Bounds (linear-scale)')
plot(range_sigma,regret_DAOL_wrtSigma,'r-', 'LineWidth', 1.5, 'DisplayName', 'DAOL', 'MarkerFaceColor', 'r');
plot(range_sigma,regret_DOCG_wrtSigma, 'b-', 'LineWidth', 1.5, 'DisplayName', 'DOCG', 'MarkerFaceColor', 'b');
plot(range_sigma,flip(regret_DOMD_wrtEpsilon), 'g-', 'LineWidth', 1.5, 'DisplayName', 'DOMD', 'MarkerFaceColor', 'g'); % We use the sigma=1-2epsilon axis
%set(gca, 'YScale', 'log');
xlabel('$\lambda_2$','Interpreter','latex');
ylabel('$\overline{\mathbf{Reg}}_j(T)$','Interpreter','latex');
legend show;
grid on;
hold off
saveas(fig1and3,strcat(path_results1and3,".fig"));

% Figure 2: Normalized Regret w.r.t. Number of iterations
range_T = cell2mat({r2{1,3,:}}); % Same range for all algorithms

regret_DAOL_wrtT = cell2mat({r2{1,1,:}}) ./ (range_T * D_def * L_def * N_def) ;
regret_DOCG_wrtT = cell2mat({r2{2,1,:}}) ./ (range_T * D_def * L_def * N_def);
regret_DOMD_wrtT = cell2mat({r2{3,1,:}}) ./ (range_T * D_def * L_def * N_def);

regret_LitDAOL_wrtT = cell2mat({r2{1,2,:}}) ./ (range_T * D_def * L_def * N_def);
regret_LitDOCG_wrtT = cell2mat({r2{2,2,:}}) ./ (range_T * D_def * L_def * N_def);
regret_LitDOMD_wrtT = cell2mat({r2{3,2,:}}) ./ (range_T * D_def * L_def * N_def);

fig2 = figure;

% Literature bounds
subplot(2,1,1);
hold on;
title('Literature Bounds (linear-scale)')
% regret_LitDOMD_wrtEpsilon and regret_DOMD_wrtEpsilon are flipped due to the mapping between sigma and epsilon
semilogy(range_T,regret_LitDAOL_wrtT,'r--o', 'LineWidth', 1.5, 'DisplayName', 'DAOL', 'MarkerFaceColor', 'r');
semilogy(range_T,regret_LitDOCG_wrtT, 'b--s', 'LineWidth', 1.5, 'DisplayName', 'DOCG', 'MarkerFaceColor', 'b');
semilogy(range_T,regret_LitDOMD_wrtT, 'g--x', 'LineWidth', 1.5, 'DisplayName', 'DOMD', 'MarkerFaceColor', 'g');
set(gca, 'YScale', 'log');
xlabel('$T$','Interpreter','latex');
ylabel('$\overline{\mathbf{Reg}}_j(T)$','Interpreter','latex');
grid on;
hold off
% PEP bounds 
subplot(2,1,2);
hold on
title('PEP Bounds')
plot(range_T,regret_DAOL_wrtT,'r-o', 'LineWidth', 1.5, 'DisplayName', 'DAOL', 'MarkerFaceColor', 'r');
plot(range_T,regret_DOCG_wrtT, 'b-s', 'LineWidth', 1.5, 'DisplayName', 'DOCG', 'MarkerFaceColor', 'b');
plot(range_T,regret_DOMD_wrtT, 'g-x', 'LineWidth', 1.5, 'DisplayName', 'DOMD', 'MarkerFaceColor', 'g'); % We use the sigma=1-2epsilon axis
%set(gca, 'YScale', 'log');
xlabel('$T$','Interpreter','latex');
ylabel('$\overline{\mathbf{Reg}}_j(T)$','Interpreter','latex');
legend show;
grid on;
hold off
saveas(fig2,strcat(path_results2,".fig"));
% Figure 4: 
range_N = cell2mat({r4{1,3,:}}); % Same range for all algorithms

% Bounds from the literature
regret_LitDAOL_wrtN = cell2mat({r4{1,2,:}}) ./ (T_def * D_def * L_def * range_N);
regret_LitDOCG_wrtN = cell2mat({r4{2,2,:}}) ./ (T_def * D_def * L_def * range_N);
regret_LitDOMD_wrtN = cell2mat({r4{3,2,:}}) ./ (T_def * D_def * L_def * range_N);

% PEP bounds
regret_DAOL_wrtN = cell2mat({r4{1,1,:}}) ./ (T_def * D_def * L_def * range_N);
regret_DOCG_wrtN = cell2mat({r4{2,1,:}}) ./ (T_def * D_def * L_def * range_N);
regret_DOMD_wrtN = cell2mat({r4{3,1,:}}) ./ (T_def * D_def * L_def * range_N);

fig4 = figure;
% Literature bounds
subplot(2,1,1);
hold on;
title('Literature Bounds (log-scale)')
% regret_LitDOMD_wrtEpsilon and regret_DOMD_wrtEpsilon are flipped due to the mapping between sigma and epsilon
semilogy(range_N,regret_LitDAOL_wrtN,'r--o', 'LineWidth', 1.5, 'DisplayName', 'DAOL', 'MarkerFaceColor', 'r');
semilogy(range_N,regret_LitDOCG_wrtN, 'b--s', 'LineWidth', 1.5, 'DisplayName', 'DOCG', 'MarkerFaceColor', 'b');
semilogy(range_N,regret_LitDOMD_wrtN, 'g--x', 'LineWidth', 1.5, 'DisplayName', 'DOMD', 'MarkerFaceColor', 'g');
set(gca, 'YScale', 'log');
xlabel('Number of Agents $N$','Interpreter','latex');
ylabel('$\overline{\mathbf{Reg}}_j(T)$','Interpreter','latex');
grid on;
hold off
% PEP bounds 
subplot(2,1,2);
hold on
title('PEP Bounds')
plot(range_N,regret_DAOL_wrtN,'r-o', 'LineWidth', 1.5, 'DisplayName', 'DAOL', 'MarkerFaceColor', 'r');
plot(range_N,regret_DOCG_wrtN, 'b-s', 'LineWidth', 1.5, 'DisplayName', 'DOCG', 'MarkerFaceColor', 'b');
plot(range_N,regret_DOMD_wrtN, 'g-x', 'LineWidth', 1.5, 'DisplayName', 'DOMD', 'MarkerFaceColor', 'g'); % We use the sigma=1-2epsilon axis
%set(gca, 'YScale', 'log');
xlabel('$N$','Interpreter','latex');
ylabel('$\overline{\mathbf{Reg}}_j(T)$','Interpreter','latex');
legend show;
grid on;
hold off

saveas(fig4,strcat(path_results4,".fig"));
