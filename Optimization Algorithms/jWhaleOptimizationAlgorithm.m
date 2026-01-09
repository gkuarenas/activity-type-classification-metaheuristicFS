function WOA = jWhaleOptimizationAlgorithm(feat, label, opts)
% Whale Optimization Algorithm with Sigmoid Binarization

% Parameters
lb = 0;
ub = 1;
b = 1; % constant

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'b'), b = opts.b; end

% Objective function
fun = @jFitnessFunction;
dim = size(feat, 2);

% Initialize whale positions
X = lb + (ub - lb) .* rand(N, dim);

% Fitness evaluation
fit = zeros(1, N);
fitG = inf;
for i = 1:N
    bin = sigmoidTransfer(X(i,:));
    fit(i) = fun(feat, label, bin, opts);
    if fit(i) < fitG
        fitG = fit(i);
        Xgb = X(i,:);
    end
end

curve = zeros(1, max_Iter);
curve(1) = fitG;
t = 2;

while t <= max_Iter
    a = 2 - t * (2 / max_Iter);  % Linearly decreasing a
    for i = 1:N
        A = 2 * a * rand() - a;
        C = 2 * rand();
        p = rand();
        l = -1 + 2 * rand();

        if p < 0.5
            if abs(A) < 1
                for d = 1:dim
                    Dx = abs(C * Xgb(d) - X(i,d));
                    X(i,d) = Xgb(d) - A * Dx;
                end
            else
                k = randi([1,N]);
                for d = 1:dim
                    Dx = abs(C * X(k,d) - X(i,d));
                    X(i,d) = X(k,d) - A * Dx;
                end
            end
        else
            for d = 1:dim
                dist = abs(Xgb(d) - X(i,d));
                X(i,d) = dist * exp(b * l) * cos(2 * pi * l) + Xgb(d);
            end
        end

        % Bound handling
        X(i,:) = max(min(X(i,:), ub), lb);
    end

    % Fitness re-evaluation
    for i = 1:N
        bin = sigmoidTransfer(X(i,:));
        fit(i) = fun(feat, label, bin, opts);
        if fit(i) < fitG
            fitG = fit(i);
            Xgb = X(i,:);
        end
    end

    curve(t) = fitG;
    fprintf('\nIteration %d Best (WOA)= %f', t, curve(t));
    t = t + 1;
end

% Final binary solution using sigmoid
final_bin = sigmoidTransfer(Xgb);
Sf = find(final_bin == 1);
sFeat = feat(:, Sf);

% Store results
WOA.sf = Sf;
WOA.ff = sFeat;
WOA.nf = length(Sf);
WOA.c  = curve;
WOA.f  = feat;
WOA.l  = label;
end

%% Sigmoid Transfer Function
function S = sigmoidTransfer(X)
    S = 1 ./ (1 + exp(-X));
    S = S > rand(size(X));  % Binary vector using stochastic rule
end
