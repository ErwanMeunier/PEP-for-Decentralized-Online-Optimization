function bound = bound_daol( T, D, L, mu, n, performance_metric, verbose, sigma)
    switch performance_metric
        case 'Individual_Regret'
            beta = sigma; % beta is the spectral gap of A
            C = (5-beta)/(1-beta); % C is defined wrt beta
            if mu > 0
                bound = 2*C*(L^2)*n*(1+log(T))/mu; % equation (6) in the paper
            else
                bound = n*(D+4*C*(L^2))*sqrt(T); % equation (7) in the paper
            end
        otherwise 
            error('This performance metric is unknown.')
    end
    if verbose 
        disp("Bound DAOL: ", bound)
    end
end