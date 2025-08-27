function bound = bound_docg(T,L,D,n,performance_metric,verbose,sigma)
    switch performance_metric
        case 'Individual_Regret'
            a = 8*n*L*D*T^(3/4);
            b = (6*sqrt(n) + 1 - sigma)*L*D*T^(3/4)/4*(sqrt(n)+1+(sqrt(n)-1)*sigma); 
            c =  2*(sqrt(n)+1+(sqrt(n)-1)*sigma)*L*D*T^(3/4)/(1-sigma) ;
            bound = a + b + c; 
        otherwise
            error('This performance metric is unknown.');
    end
    if verbose
        disp("Bound DOCG: ",bound);
    end
end