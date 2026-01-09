% [2016] - "Jaya: A simple and new optimization algorithm for solving 
% constrained and unconstrained optimization problems"
% Modified to use sigmoid transfer function (probabilistic binarization)

function JA = jJayaAlgorithm(feat, label, opts)
% Parameters
lb = 0;
ub = 1;

if isfield(opts, 'N'), N = opts.N; end
if isfield(opts, 'T'), max_Iter = opts.T; end

% Objective function
fun = @jFitnessFunction;

% Number of dimensions
dim = size(feat, 2);

% Initial population
X = zeros(N, dim);
for i = 1:N
    for d = 1:dim
        X(i, d) = lb + (ub - lb) * rand();
    end
end

% Fitness initialization
fit = zeros(1, N);
fitG = inf;

for i = 1:N
    % Apply sigmoid and probabilistic binarization
    prob = 1 ./ (1 + exp(-X(i, :)));
    bin = rand(1, dim) < prob;

    fit(i) = fun(feat, label, bin, opts);

    if fit(i) < fitG
        fitG = fit(i);
        Xgb = X(i, :);
    end
end

% Preallocate
Xnew = zeros(N, dim);
curve = zeros(1, max_Iter);
curve(1) = fitG;
t = 2;

% Main loop
while t <= max_Iter
    [~, idxB] = min(fit);
    Xbest = X(idxB, :);

    [~, idxW] = max(fit);
    Xworst = X(idxW, :);

    for i = 1:N
        for d = 1:dim
            r1 = rand();
            r2 = rand();

            % Jaya update with abs (per original paper)
            Xnew(i, d) = X(i, d) + ...
                r1 * (Xbest(d) - abs(X(i, d))) - ...
                r2 * (Xworst(d) - abs(X(i, d)));
        end

        % Boundary check
        XB = Xnew(i, :);
        XB(XB > ub) = ub;
        XB(XB < lb) = lb;
        Xnew(i, :) = XB;
    end

    for i = 1:N
        prob = 1 ./ (1 + exp(-Xnew(i, :)));
        bin = rand(1, dim) < prob;

        Fnew = fun(feat, label, bin, opts);

        if Fnew < fit(i)
            fit(i) = Fnew;
            X(i, :) = Xnew(i, :);
        end

        if fit(i) < fitG
            fitG = fit(i);
            Xgb = X(i, :);
        end
    end

    curve(t) = fitG;
    fprintf('\nIteration %d Best (JA)= %f', t, curve(t));
    t = t + 1;
end

% Final binarization
prob_final = 1 ./ (1 + exp(-Xgb));
bin_final = rand(1, dim) < prob_final;

Pos = 1:dim;
Sf = Pos(bin_final == 1);
sFeat = feat(:, Sf);

% Output
JA.sf = Sf;
JA.ff = sFeat;
JA.nf = length(Sf);
JA.c = curve;
JA.f = feat;
JA.l = label;
end
