%%%%%%%%%% Optimize step-sizes for Distributed Online Conditional Gradient (DOCG) %%%%%%%%%
% 0 <= a <= D
% 0 <= b <= L
% 0 <= c <= 1

function [best_step_sizes, pep_bound] = design_DOCG(T_LB, T_UB, initial_point_T_LB, sigma)
    best_step_sizes = cell(1, T_UB-T_LB+1);
    pep_bound = cell(1, T_UB-T_LB+1);

    % First run at lower bound
    [best_step_sizes{1}, pep_bound{1}] = design_DOCG_for_T(T_LB, initial_point_T_LB, sigma);

    % Warm-start optimization for subsequent T values
    for i = 2:(T_UB-T_LB+1)
        T = T_LB + (i-1);
        [best_step_sizes{i}, pep_bound{i}] = design_DOCG_for_T(T, best_step_sizes{i-1}, sigma);
    end
end


function [x, fval] = design_DOCG_for_T(T, initial_point, sigma)
    % Default parameters (MUST match with simulations)
    L = 1;
    D = 1;
    n = 2;   % number of agents
    performance_metric = 'Individual_Regret'; % or 'Averaged_Individual_Regret'
    verbose = 0;

    % Optimization bounds
    lb = [0, 0, 0];
    ub = [D, L, 1];

    % Surrogate optimization setup
    options = optimoptions('surrogateopt', ...
                           'PlotFcn', 'surrogateoptplot', ...
                           'MaxFunctionEvaluations', 50, ...
                           'InitialPoints', initial_point);

    [x, fval] = surrogateopt(@objconstr, lb, ub, options);

    % Objective function wrapper
    function wc = objconstr(param)
        a = param(1);
        b = param(2);
        c = param(3);

        compute_step_size = @(t) a / (t^c + b);

        wc = distributed_online_conditional_gradient_given_step_sizes( ...
                 T, D, L, n, performance_metric, verbose, sigma, compute_step_size);
    end
end
