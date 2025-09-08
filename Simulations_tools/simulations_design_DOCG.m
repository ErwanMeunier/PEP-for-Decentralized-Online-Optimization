OUTPUT_DIR = "./Simulation_Results";
rng default; % fixing a seed for the random generator

% Default parameters
T_LB = 15;
T_UB = 15;
L = 1;
D = 1;
n = 2;
performance_metric = 'Individual_Regret'; % or 'Averaged_Individual_Regret'
verbose = 0;

%%%%%%%%%%%%%%%%%%%
sim_opt_SS = true;   % whether to optimize step-sizes
plot_figbyfig = false;
plot_together = false;
plot_wrt_sigma = true;
% THE MARKER ARRAY MUST BE DEFINED ACCORDING TO THE LENGTH OF RANGE_SIGMA
%range_sigma = [0.1,0.25,0.5,0.75,0.9];

range_sigma = linspace(0,0.9,10);

bounds_to_plot = cell(length(range_sigma),2,T_UB - T_LB + 1);

for i = 1:length(range_sigma)
    sigma = range_sigma(i);

    if sim_opt_SS
        resultsDesign = cell(4,T_UB - T_LB + 1);

        % Convention: 
        % 1 --> T number of iterations
        % 2 --> Loose bound from Literature
        % 3 --> PEP bound with standard step-size
        % 4 --> PEP bound with optimized step-size  

        initial_point_T_LB = [(1-sigma)*D/(2*(sqrt(n)+1+(sqrt(n)-1)*sigma)*T^(3/4)),L,0];% [1/2, 0, 1/2]; % Standard step-size
        
        [best_step_sizes_global, pep_bound_global] = design_DOCG(T_LB, T_UB, initial_point_T_LB, sigma);

        % Filling results
        for k = 1:(T_UB - T_LB + 1)
            T = T_LB + k - 1;
            resultsDesign{1,k} = T;

            % Replace with your theoretical bound function if available
            resultsDesign{2,k} = bound_docg(T, D, L, n, performance_metric, verbose, sigma);

            % PEP bound with standard step-size
            resultsDesign{3,k} = distributed_online_conditional_gradient(T, D, L, n, performance_metric, verbose, sigma);

            % PEP bound with optimized step-size
            resultsDesign{4,k} = pep_bound_global{k};
        end
    end

    %%%%%%%%%%%%%%%%% SAVING RESULTS %%%%%%%%%%%%%%%%%
    path_result_design = fullfile(OUTPUT_DIR, sprintf("result_designDOCGT_LB%dT_UB%dSIGMA%g.csv", T_LB, T_UB, sigma));
    path_best_step_sizes = fullfile(OUTPUT_DIR, sprintf("best_step_sizes_designDOCGT_LB%dT_UB%dSIGMA%g.csv", T_LB, T_UB, sigma));

    fprintf("best_step_sizes_designDOCGT_LB%dT_UB%dSIGMA%g.csv \n", T_LB, T_UB, sigma);

    if sim_opt_SS
        % Creating the files
        for filename = [path_result_design, path_best_step_sizes]
            fprintf("Creating %s \n", filename);
            fid = fopen(filename,'w'); fclose(fid);
        end

        % Writing the data
        writecell(resultsDesign, path_result_design, 'FileType','text');
        writecell(best_step_sizes_global, path_best_step_sizes, 'FileType','text');
    end

    display("here")
    %%%%%%%%%%%%%%%%% POST PROCESSING and PLOTTING %%%%%%%%%%%%%%%%
    res_design_2D = readcell(path_result_design);
    res_design_linear = res_design_2D(:); 
    res_design = reshape(res_design_linear, [4, T_UB - T_LB + 1]);

    % Normalizing the regret
    regret_LitBound       = cell2mat({res_design{2,:}}) ./ ((T_LB:T_UB) * L * D * n);
    regret_PEPBoundLitSS  = cell2mat({res_design{3,:}}) ./ ((T_LB:T_UB) * L * D * n);
    regret_PEPBoundOptSS  = cell2mat({res_design{4,:}}) ./ ((T_LB:T_UB) * L * D * n);

    if plot_figbyfig 
        fig_design = figure;
        hold on;
            title("Performance of DOCG with improved bounds");
            plot(T_LB:T_UB, regret_PEPBoundLitSS, 'b-s', 'LineWidth', 1.5, 'DisplayName', 'PEP: Lit. step-sizes', 'MarkerFaceColor','b');
            plot(T_LB:T_UB, regret_PEPBoundOptSS, 'g-x', 'LineWidth', 1.5, 'DisplayName', 'PEP: Improved step-sizes', 'MarkerFaceColor','g');
            xticks(T_LB:T_UB)
            xlabel("$T$", 'Interpreter','latex');
            ylabel('$\overline{\mathbf{Reg}}_j(T)$','Interpreter','latex');
            legend show;
            grid on;
        hold off;
        saveas(fig_design, strcat(path_result_design, ".fig"));
    end

    if plot_together || plot_wrt_sigma
        bounds_to_plot(i,1,:) = num2cell(regret_PEPBoundLitSS);
        bounds_to_plot(i,2,:) = num2cell(regret_PEPBoundOptSS);
    end
end

% Once data have been parsed
if plot_together 
    marker = ["-<","-^","->","-v","-diamond"];
    path_results_design_together = fullfile(OUTPUT_DIR, "final_figure_design_DOCG");
    fig_design_together = figure; 
    hold on;
    title("Worst-case bound: Literature vs Optimized Step-Sizes")

    color = parula(length(range_sigma));
    for i=1:length(range_sigma)
        improvement = 100 * (1 - cell2mat({bounds_to_plot{i,2,:}}) ./ cell2mat({bounds_to_plot{i,1,:}}));
        legend_text = num2str(range_sigma(i));
        plot(T_LB:T_UB, improvement, marker(i), 'LineWidth',1.5, ...
             'Color', color(i,:), 'MarkerEdgeColor', color(i,:), 'MarkerFaceColor', color(i,:));
    end
    lgd = legend(num2str(range_sigma(:)), 'Location', 'northwest');
    title(lgd, "$\lambda_2$",'Interpreter','Latex');
    xlabel("$T$",'Interpreter','latex');
    ylabel('Improvement of the PEP bound (%)');
    grid on
    hold off;
    saveas(fig_design_together, strcat(path_results_design_together, ".fig"));
end

if plot_wrt_sigma
    path_results_sigma_together = fullfile(OUTPUT_DIR, "final_figure_improvement_vs_sigma");
    fig_sigma_together = figure;
    hold on;
    title("Improvement vs \lambda_2 for different T")

    color = parula(T_UB - T_LB + 1);
    marker = ["-<","-^","->","-v","-diamond","-o","-s","-*","-p"];
    for k = 1:(T_UB - T_LB + 1)
        T = T_LB + k - 1;
        imp_vs_sigma = zeros(1, length(range_sigma));
        for i = 1:length(range_sigma)
            imp_vs_sigma(i) = 100 * (1 - bounds_to_plot{i,2,k} ./ bounds_to_plot{i,1,k});
        end
        plot(range_sigma, imp_vs_sigma, marker(mod(k-1,length(marker))+1), ...
             'LineWidth',1.5, 'Color', color(k,:), ...
             'DisplayName', sprintf('T = %d', T));
    end
    xlabel("$\lambda_2$",'Interpreter','latex');
    ylabel('Improvement of the PEP bound (%)');
    legend('Location','northwest');
    grid on;
    hold off;
    saveas(fig_sigma_together, strcat(path_results_sigma_together, ".fig"));
end
