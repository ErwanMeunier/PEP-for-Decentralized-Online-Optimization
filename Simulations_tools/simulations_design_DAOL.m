OUTPUT_DIR = "./Simulation_Results";
rng default; % fixing a seed for the random generator
% Default parameters
T_LB = 1;
T_UB = 10;
L_DAOL = 1;
D_DAOL = 1;
N_DAOL = 2;
MU_DAOL = 0;

%%%%%%%%%%%%%%%%%%%
sim_opt_SS = false;
plot_figbyfig = false;
plot_together = true;
range_SIGMA =  [0.1,0.25,0.5,0.75,0.9];
bounds_to_plot = cell(length(range_SIGMA),2,T_UB - T_LB +1);
for i=1:length(range_SIGMA)
    SIGMA_DAOL = range_SIGMA(i);
    if sim_opt_SS
        resultsDesign = cell(4,T_UB - T_LB + 1);
    
        % Convention: 1 --> T Number of iterations,
        %  2 --> Loose bound from Literature,
        %  3 --> PEP bound standard step-size,
        %  4 --> PEP bound optimized step-size  
    
        initial_point_T_LB = [1/2, 0, 1/2]; % Standard step-size
        [best_step_sizes_global, pep_bound_global] = design_DAOL(T_LB,T_UB,initial_point_T_LB);
        % best_step_sizes_global will be saved apart...
        % filling for 1, 2 and 3 and 4
        for k=1:(T_UB-T_LB+1)
            T = T_LB + i-1;
            resultsDesign{1,k} = T;
            % GO TO DESIGN DAOL FOR THE VALUE OF PARAMETERS
            resultsDesign{2,k} = bound_daol(T, 1, 1, 0, 2, 'Individual_Regret', 0, SIGMA_DAOL);
            resultsDesign{3,k} = distributed_autonomous_online_learning(T, 1, 1, 0, 2, 'Individual_Regret', 0, SIGMA_DAOL);
            resultsDesign{4,k} = pep_bound_global{k};
        end
    end
    %%%%%%%%%%%%%%%%% SAVING RESULTS %%%%%%%%%%%%%%%%%
    
    path_result_design = fullfile(OUTPUT_DIR,sprintf("result_designDAOLT_LB%dT_UB%dSIGMA%d.csv",T_LB,T_UB,SIGMA_DAOL));
    path_best_step_sizes = fullfile(OUTPUT_DIR,sprintf("best_step_sizes_designDAOLT_LB%dT_UB%dSIGMA%d.csv",T_LB,T_UB,SIGMA_DAOL));
    fprintf("best_step_sizes_designDAOLT_LB%dT_UB%dSIGMA%d.csv \n",T_LB,T_UB,SIGMA_DAOL);
    if sim_opt_SS
        % Creating the file
        for filename = [path_result_design,path_best_step_sizes]
            fprintf("Creating %s \n",filename);
            fid = fopen(filename,'w');
            fprintf(filename,'');         
            fid = fclose(fid);
        end
    
        % Writing the data
        writecell(resultsDesign,path_result_design,'FileType','text');
        writecell(best_step_sizes_global,path_best_step_sizes,'FileType','text');
    end
    %%%%%%%%%%%%%%%%% POST PROCESSING and PLOTTING %%%%%%%%%%%%%%%%
    
    % Importing results
    res_design_2D = readcell(path_result_design);

    % Flattening
    res_design_linear = res_design_2D(:); 

    % Reshaping the cell array
    res_design = reshape(res_design_linear,[4,T_UB-T_LB+1]);

    % Normalizing the regret
    regret_LitBound = cell2mat({res_design{2,:}}) ./ (T_LB:T_UB)*L_DAOL*D_DAOL*N_DAOL;
    regret_PEPBoundLitSS = cell2mat({res_design{3,:}}) ./ (T_LB:T_UB)*L_DAOL*D_DAOL*N_DAOL;
    regret_PEPBoundOptSS = cell2mat({res_design{4,:}}) ./ (T_LB:T_UB)*L_DAOL*D_DAOL*N_DAOL;
    
    if plot_figbyfig 
        % Plotting 
        fig_design = figure;
        hold on;
            title("Performance of DAOL with improved bounds");
            %plot(T_LB:T_UB,regret_LitBound, 'r--o', 'LineWidth', 1.5, 'DisplayName', 'Literature Bound', 'MarkerFaceColor', 'r');
            plot(T_LB:T_UB,regret_PEPBoundLitSS, 'b-s', 'LineWidth', 1.5, 'DisplayName', 'PEP: Lit. step-sizes', 'MarkerFaceColor', 'b');
            plot(T_LB:T_UB,regret_PEPBoundOptSS, 'g-x', 'LineWidth', 1.5, 'DisplayName', 'PEP: Improved step-sizes', 'MarkerFaceColor', 'g');
            xticks(T_LB:T_UB)
            xlabel("$T$", 'Interpreter', 'latex');
            ylabel('$\overline{\mathbf{Reg}}_j(T)$','Interpreter','latex');
            legend show;
            grid on;
        hold off;
        % Saving the figure
        saveas(fig_design,strcat(path_result_design,".fig"))
    end
    if plot_together % Getting the parsed data
        %display(num2cell(regret_PEPBoundLitSS));
        %display(num2cell(regret_PEPBoundOptSS));
        bounds_to_plot(i,1,:) = num2cell(regret_PEPBoundLitSS);
        bounds_to_plot(i,2,:) = num2cell(regret_PEPBoundOptSS);
    end 
end
% Once data have been parsed

if plot_together 
    marker = ["-<","-^","->","-v"];
    path_results_design_together = fullfile(OUTPUT_DIR, "final_figure_design");
    fig_design_together = figure; 
    hold on;
    title("Worst-case bound Literature v.s. Optimized Step-Sizes")
    
    color = parula(length(range_SIGMA));
    for i=1:length(range_SIGMA)
        %display(cell2mat({bounds_to_plot{i,2,:}}));
        display(cell2mat({bounds_to_plot{i,2,:}}) ./ cell2mat({bounds_to_plot{i,1,:}}));
        legend_text = num2str(range_SIGMA(i));
        p = plot(T_LB:T_UB, 100*(1 - cell2mat({bounds_to_plot{i,2,:}}) ./ cell2mat({bounds_to_plot{i,1,:}})),...
             marker(i) , 'LineWidth', 1.5, 'Color', color(i,:), 'MarkerEdgeColor', color(i,:), 'MarkerFaceColor', color(i,:));
    end
    lgd = legend(num2str(range_SIGMA(:)), 'Location', 'northwest');
    title(lgd, '\lambda_2');
    xlabel("$T$",'Interpreter','latex');
    ylabel('Improvement of the PEP bound (%)');
    grid on
    %lgd show;
    hold off;
    saveas(fig_design_together,strcat(path_results_design_together,".fig"));
end


