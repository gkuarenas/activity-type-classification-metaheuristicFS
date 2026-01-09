function GWO = jGreyWolfOptimizer(feat, label, opts)
% Grey Wolf Optimizer (Binary with Sigmoid Transfer)
% feat: features matrix (nSamples x nFeatures)
% label: target labels
% opts: structure with fields:
%   - N: number of wolves (population size)
%   - T: max iterations
%   - thres: (optional) ignored, replaced by sigmoid

% --- Parameters
lb = 0;
ub = 1;

if isfield(opts, 'N'), N = opts.N; else, N = 20; end
if isfield(opts, 'T'), max_Iter = opts.T; else, max_Iter = 100; end

fun = @jFitnessFunction;  % external fitness function
dim = size(feat, 2);      % number of features

% --- Initialization
X = lb + (ub - lb) * rand(N, dim);  % position matrix
fit = zeros(1, N);                  % fitness values

% Evaluate initial wolves using sigmoid binarization
for i = 1:N
    S = 1 ./ (1 + exp(-X(i, :)));            % sigmoid
    bin = rand(1, dim) < S;                  % probabilistic threshold
    fit(i) = fun(feat, label, bin, opts);    % fitness
end

% Sort and identify alpha, beta, delta
[~, idx] = sort(fit, 'ascend');
Xalpha = X(idx(1), :);
Xbeta  = X(idx(2), :);
Xdelta = X(idx(3), :);
Falpha = fit(idx(1));
Fbeta  = fit(idx(2));
Fdelta = fit(idx(3));

curve = zeros(1, max_Iter);  % convergence curve
curve(1) = Falpha;
t = 2;

% --- Main loop
while t <= max_Iter
    a = 2 - t * (2 / max_Iter);  % linearly decreasing a

    for i = 1:N
        for d = 1:dim
            % Coefficients
            C1 = 2 * rand();
            C2 = 2 * rand();
            C3 = 2 * rand();
            A1 = 2 * a * rand() - a;
            A2 = 2 * a * rand() - a;
            A3 = 2 * a * rand() - a;

            % Distances to leaders
            Dalpha = abs(C1 * Xalpha(d) - X(i,d));
            Dbeta  = abs(C2 * Xbeta(d)  - X(i,d));
            Ddelta = abs(C3 * Xdelta(d) - X(i,d));

            % New positions toward leaders
            X1 = Xalpha(d) - A1 * Dalpha;
            X2 = Xbeta(d)  - A2 * Dbeta;
            X3 = Xdelta(d) - A3 * Ddelta;

            % Update position
            X(i,d) = (X1 + X2 + X3) / 3;
        end

        % Boundary control
        X(i, X(i,:) > ub) = ub;
        X(i, X(i,:) < lb) = lb;
    end

    % Fitness evaluation
    for i = 1:N
        S = 1 ./ (1 + exp(-X(i,:)));   % sigmoid
        bin = rand(1, dim) < S;       % binary conversion
        fit(i) = fun(feat, label, bin, opts);

        % Update leaders
        if fit(i) < Falpha
            Falpha = fit(i);
            Xalpha = X(i,:);
        elseif fit(i) < Fbeta && fit(i) > Falpha
            Fbeta = fit(i);
            Xbeta = X(i,:);
        elseif fit(i) < Fdelta && fit(i) > Falpha && fit(i) > Fbeta
            Fdelta = fit(i);
            Xdelta = X(i,:);
        end
    end

    curve(t) = Falpha;
    fprintf('\nIteration %d Best (GWO)= %f', t, curve(t));
    t = t + 1;
end

% Final binarization using best wolf (Xalpha)
Sfinal = 1 ./ (1 + exp(-Xalpha));
final_bin = rand(1, dim) < Sfinal;
Sf = find(final_bin);         % selected feature indices
sFeat = feat(:, Sf);          % reduced feature set

% Store results
GWO.sf = Sf; 
GWO.ff = sFeat;
GWO.nf = length(Sf);
GWO.c  = curve;
GWO.f  = feat;
GWO.l  = label;
end
