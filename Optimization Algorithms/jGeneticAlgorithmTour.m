function GA = jGeneticAlgorithmTour(feat, label, opts)
% Genetic Algorithm with Sigmoid Transfer Function

% Parameters
CR        = 0.8;   % Crossover rate
MR        = 0.01;  % Mutation rate
Tour_size = 3;     % Tournament size

if isfield(opts, 'N'), N = opts.N; end
if isfield(opts, 'T'), max_Iter = opts.T; end
if isfield(opts, 'CR'), CR = opts.CR; end
if isfield(opts, 'MR'), MR = opts.MR; end
if isfield(opts, 'Ts'), Tour_size = opts.Ts; end

% Objective function
fun = @jFitnessFunction;

% Number of dimensions
dim = size(feat, 2);

% Initial population (real values in [-1,1])
X = jInitialization(N, dim);

% Fitness
fit  = zeros(1, N);
fitG = inf;

for i = 1:N
    bin_X = sigmoid(X(i,:)) > 0.5;
    fit(i) = fun(feat, label, bin_X, opts);
    if fit(i) < fitG
        fitG = fit(i);
        Xgb  = X(i,:);
    end
end

curve = zeros(1, max_Iter);
curve(1) = fitG;
t = 2;

% Main loop
while t <= max_Iter
    Xc1 = zeros(0, dim);
    Xc2 = zeros(0, dim);
    fitC1 = [];
    fitC2 = [];
    z = 1;

    for i = 1:N
        if rand() < CR
            % Tournament selection
            k1 = jTournamentSelection(fit, Tour_size, N);
            k2 = jTournamentSelection(fit, Tour_size, N);
            P1 = X(k1,:);
            P2 = X(k2,:);

            % Single-point crossover
            ind = randi([1, dim - 1]);
            child1 = [P1(1:ind), P2(ind + 1:end)];
            child2 = [P2(1:ind), P1(ind + 1:end)];

            % Mutation
            for d = 1:dim
                if rand() < MR
                    child1(d) = rand()*2 - 1; % reinitialize in [-1,1]
                end
                if rand() < MR
                    child2(d) = rand()*2 - 1;
                end
            end

            % Sigmoid transfer + binarization
            bin1 = sigmoid(child1) > 0.5;
            bin2 = sigmoid(child2) > 0.5;

            % Evaluate fitness
            fitC1(1, z) = fun(feat, label, bin1, opts);
            fitC2(1, z) = fun(feat, label, bin2, opts);

            % Save children
            Xc1(z,:) = child1;
            Xc2(z,:) = child2;
            z = z + 1;
        end
    end

    % Merge population
    XX = [X; Xc1; Xc2];
    FF = [fit, fitC1, fitC2];

    % Select best N
    [FF, idx] = sort(FF, 'ascend');
    X = XX(idx(1:N), :);
    fit = FF(1:N);

    % Update best
    if fit(1) < fitG
        fitG = fit(1);
        Xgb  = X(1,:);
    end

    curve(t) = fitG;
    fprintf('\nGeneration %d Best (GA Sigmoid) = %f', t, curve(t));
    t = t + 1;
end

% Select final features
binFinal = sigmoid(Xgb) > 0.5;
Sf = find(binFinal == 1);
sFeat = feat(:, Sf);

% Output
GA.sf = Sf;
GA.ff = sFeat;
GA.nf = length(Sf);
GA.c  = curve;
GA.f  = feat;
GA.l  = label;
end

%--------------------------------------------

function Index = jTournamentSelection(fit, Tour_size, N)
Tour_idx  = randsample(N, Tour_size);
Tour_fit  = fit(Tour_idx);
[~, idx]  = min(Tour_fit);
Index     = Tour_idx(idx);
end

%--------------------------------------------

function X = jInitialization(N, dim)
X = -1 + 2 * rand(N, dim);  % real values in [-1, 1]
end

%--------------------------------------------

function y = sigmoid(x)
y = 1 ./ (1 + exp(-x));
end
