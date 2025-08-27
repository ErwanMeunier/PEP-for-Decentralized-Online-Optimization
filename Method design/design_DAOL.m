%%%%%%%%%% Here we will minimize the worst-case regret of DAOL by fine-tuning step-sizes %%%%%%%%%
% 0 <= a <= D_DAOL
% 0 <= b <= L
% 0 <= c <= 1

function [best_step_sizes,pep_bound]=design_DAOL(T_LB,T_UB,initial_point_T_LB,SIGMA_DAOL)
    best_step_sizes = cell(1,T_UB-T_LB+1);
    pep_bound = cell(1,T_UB-T_LB+1);
    [best_step_sizes{1}, pep_bound{1}] = design_DAOL_for_T(T_LB,initial_point_T_LB,SIGMA_DAOL);
    for i=2:(T_UB-T_LB+1)
        T = T_LB + (i-1);
        [best_step_sizes{i}, pep_bound{i}] = design_DAOL_for_T(T,best_step_sizes{i-1},SIGMA_DAOL);
    end
end 

function [x,fval]=design_DAOL_for_T(T,initial_point,SIGMA_DAOL)
    % DEFAULT PARAMETERS MUST BE THE SAME THAN IN SIMULATIONS_DESIGN.M
    L_DAOL = 1;
    D_DAOL = 1;
    N_DAOL = 2;
    MU_DAOL = 0;

    lb = [0, 0, 0];
    ub = [D_DAOL, L_DAOL, 1];
    options = optimoptions('surrogateopt','PlotFcn','surrogateoptplot', ...
                            'MaxFunctionEvaluations',50,'InitialPoints',initial_point);
    
    [x,fval] = surrogateopt(@objconstr,lb,ub,options);

    function [wc]=objconstr(param)
        %display(param);
        a = param(1);
        b = param(2);
        c = param(3);
        compute_step_size = @(t) a/(t^c + b);
        wc = distributed_autonomous_online_learning_given_step_sizes(T,D_DAOL,L_DAOL,MU_DAOL,N_DAOL,'Individual_Regret',0,SIGMA_DAOL,compute_step_size);
    end
end

