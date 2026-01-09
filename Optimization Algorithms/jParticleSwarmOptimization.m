function PSO = jParticleSwarmOptimization(feat, label, opts)
% Binary Particle Swarm Optimization with dynamic inertia & early stopping

% Parameters
lb    = 0;
ub    = 1;
c1    = 2;
c2    = 2;
w_max = 0.9;
w_min = 0.4;
Vmax  = (ub - lb) / 2;

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'c1'), c1 = opts.c1; end 
if isfield(opts,'c2'), c2 = opts.c2; end 
if isfield(opts,'w_max'), w_max = opts.w_max; end 
if isfield(opts,'w_min'), w_min = opts.w_min; end 
if isfield(opts,'Vmax'), Vmax = opts.Vmax; end

% Objective function
fun = @jFitnessFunction;
dim = size(feat, 2);

% Initialize population
X = rand(N, dim);
V = zeros(N, dim);
fit  = zeros(1, N);
fitG = inf;

% Sigmoid transfer function
sigmoid = @(v) 1 ./ (1 + exp(-v));

% Evaluate initial particles
for i = 1:N
    bin = rand(1, dim) < sigmoid(X(i,:));
    fit(i) = fun(feat, label, bin, opts);
    
    if fit(i) < fitG
        Xgb  = X(i,:);
        fitG = fit(i);
    end
end

Xpb  = X;
fitP = fit;

% Convergence tracking
curve = zeros(1, max_Iter);
curve(1) = fitG;
no_improve_counter = 0;

t = 2;
while t <= max_Iter
    w = w_max - ((w_max - w_min) * (t - 1) / (max_Iter - 1));
    
    prev_best = fitG;
    
    for i = 1:N
        for d = 1:dim
            r1 = rand();
            r2 = rand();
            V(i,d) = w * V(i,d) + ...
                     c1 * r1 * (Xpb(i,d) - X(i,d)) + ...
                     c2 * r2 * (Xgb(d) - X(i,d));
            V(i,d) = max(min(V(i,d), Vmax), -Vmax);
            S = sigmoid(V(i,d));
            X(i,d) = rand() < S;
        end

        bin = X(i,:) > 0.5;
        fit(i) = fun(feat, label, bin, opts);
        
        if fit(i) < fitP(i)
            Xpb(i,:) = X(i,:);
            fitP(i)  = fit(i);
        end
        
        if fitP(i) < fitG
            Xgb  = Xpb(i,:);
            fitG = fitP(i);
        end
    end

    curve(t) = fitG;
    fprintf('\nIteration %d Best (BPSO) = %f', t, fitG);
    
    % Early stopping check
    if abs(prev_best - fitG) < 1e-6
        no_improve_counter = no_improve_counter + 1;
    else
        no_improve_counter = 0;
    end

    t = t + 1;
end

% Final binary selection
binFinal = Xgb > 0.5;
Sf       = find(binFinal == 1);
sFeat    = feat(:, Sf);

% Output
PSO.sf = Sf;
PSO.ff = sFeat;
PSO.nf = length(Sf);
PSO.c  = curve;
PSO.f  = feat;
PSO.l  = label;
end
