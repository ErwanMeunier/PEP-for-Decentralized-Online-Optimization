% Distributed mirror descent experiment
test_domd = @(T,G,lambda,n,sigma) distributed_mirror_descent_online_optimization( ...
    T, 1, 1, G, lambda, n, 'Individual_Regret', 0, sigma, 1e-6) / (T * n);

n = 2;
T = 5;
NB_KAPPA = 6;              % number of condition number samples (excluding +Inf)
NB_SAMPLES_SIGMA = 10;     % number of sigma samples
result = zeros(NB_KAPPA+1, NB_SAMPLES_SIGMA);
kappa_vect = zeros(NB_KAPPA+1, 1);   % condition numbers
sigma_vect = linspace(0, 1, NB_SAMPLES_SIGMA);

lambda = 0.999;  % fix strong convexity

% Loop over condition numbers by varying G
for k = 1:NB_KAPPA
    G = 10^(k-1);
    kappa_vect(k) = G / lambda;   % condition number
    fprintf("k=%d\n",k);
    for k_sigma = 1:NB_SAMPLES_SIGMA
        fprintf("k_sigma=%d\n",k_sigma);
        result(k, k_sigma) = test_domd(T, G, lambda, n, sigma_vect(k_sigma));
    end
end

% Special case: infinite condition number (G->+Inf)
kappa_vect(NB_KAPPA+1) = Inf;
for k_sigma = 1:NB_SAMPLES_SIGMA
    result(NB_KAPPA+1, k_sigma) = test_domd(T, Inf, 0.99, n, sigma_vect(k_sigma));
end

% --- Plotting ---
figure; hold on;
colors = parula(NB_KAPPA+1);

for k = 1:NB_KAPPA
    plot(sigma_vect, result(k, :), '-o', ...
        'DisplayName', sprintf('$\\kappa = 10^{%d}$', k-1), ...
        'Color', colors(k,:));
end

plot(sigma_vect, result(NB_KAPPA+1, :), '-o', ...
    'DisplayName', '$\kappa = +\infty$', ...
    'Color', colors(NB_KAPPA+1,:));

xlabel('$\lambda_2$', 'Interpreter', 'latex');
ylabel('$\overline{\mathbf{Reg}}_j(T)$', 'Interpreter', 'latex');
title('Scaled Regret for various condition numbers of the Kernel', 'Interpreter', 'latex');
lgd = legend('Interpreter', 'latex', 'Location', 'best');

lgd.Title.String = 'Condition Number $\kappa=\frac{G}{\mu}$ of the Kernel';
lgd.Title.Interpreter = 'latex';

grid on;
