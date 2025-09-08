% xi: bound over the  
function [bound]=bound_DOMD(T,D,L,G,lambda,n,performance_metric,verbose,sum_norm_first_estimates,sum_Bregman_first_estimates,eta,epsilon)
    xi = epsilon;
    B = 1;
    switch performance_metric
        case 'Individual_Regret'
            nu = (1-xi/(4*n^2))^(-2);
            kappa = (1-xi/(4*n^2))^(1/B);
            A0 = 2*nu*L*sum_norm_first_estimates/(1-kappa); 
            A1 = sum_Bregman_first_estimates;
            A2 = (n/lambda)*(L^2 /2 + (2*(nu)/(1-kappa)) * (L^2));
            %A3 = n*sqrt(2/lambda)*(L+2*(nu/(1-kappa))*L); % Useless since we assume rho_t=0 for all t 
            %A4 = 2*n*sqrt(2/lambda)*G*D; % Useless since we assume rho_t=0 for all t 
            bound = T*(A0/T + A1 / (eta*T) + A2*eta);
        otherwise
        error("Unknown performance metric.")
    end
    if verbose
        fprintf("bound_DOMD=%d",bound);
    end
end