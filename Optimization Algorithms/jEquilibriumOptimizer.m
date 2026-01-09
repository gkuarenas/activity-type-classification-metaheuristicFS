function EO = jEquilibriumOptimizer(feat,label,opts)
% Equilibrium Optimizer with sigmoid transfer function

% Parameters
lb    = 0;
ub    = 1; 
a1    = 2;     
a2    = 1;     
GP    = 0.5;   
V     = 1;     

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'a1'), a1 = opts.a1; end  
if isfield(opts,'a2'), a2 = opts.a2; end  
if isfield(opts,'GP'), GP = opts.GP; end  

fun = @jFitnessFunction; 
dim = size(feat,2); 

% Initialization
X   = lb + (ub - lb) .* rand(N, dim); 
Xmb = X;
fit = inf(1,N); 
fitM = inf(1,N); 

% Equilibrium candidates
fitE1 = inf; fitE2 = inf; fitE3 = inf; fitE4 = inf;
Xeq1 = zeros(1,dim); Xeq2 = zeros(1,dim); Xeq3 = zeros(1,dim); Xeq4 = zeros(1,dim);
curve = inf(1,max_Iter); 
t = 1;

while t <= max_Iter
    % Evaluate fitness with sigmoid transfer
    for i = 1:N
        bin = sigmoidTransfer(X(i,:));
        fit(i) = fun(feat, label, bin, opts);
        
        % Update equilibrium candidates
        if fit(i) < fitE1
            [fitE1, Xeq1] = deal(fit(i), X(i,:));
        elseif fit(i) < fitE2
            [fitE2, Xeq2] = deal(fit(i), X(i,:));
        elseif fit(i) < fitE3
            [fitE3, Xeq3] = deal(fit(i), X(i,:));
        elseif fit(i) < fitE4
            [fitE4, Xeq4] = deal(fit(i), X(i,:));
        end
    end
    
    % Memory update
    for i = 1:N
        if fitM(i) < fit(i)
            fit(i) = fitM(i);
            X(i,:) = Xmb(i,:);
        end
    end
    Xmb = X; fitM = fit;

    % Average equilibrium
    Xave = (Xeq1 + Xeq2 + Xeq3 + Xeq4) / 4;
    Xpool = [Xeq1; Xeq2; Xeq3; Xeq4; Xave];
    
    % Time function
    T = (1 - (t / max_Iter)) ^ (a2 * (t / max_Iter));
    
    % Update positions
    for i = 1:N
        r1 = rand(); r2 = rand();
        GCP = 0.5 * r1 * (r2 < GP);
        eq = randi([1,5]);
        
        for d = 1:dim
            r = rand();
            lambda = rand();
            F = a1 * sign(r - 0.5) * (exp(-lambda * T) - 1);
            G0 = GCP * (Xpool(eq,d) - lambda * X(i,d));
            G = G0 * F;
            
            X(i,d) = Xpool(eq,d) + (X(i,d) - Xpool(eq,d)) * F + ...
                     (G / (lambda * V)) * (1 - F);
        end
        % Bound check
        X(i,:) = max(min(X(i,:), ub), lb);
    end

    curve(t) = fitE1;
    fprintf('\nIteration %d Best (EO)= %f', t, curve(t));
    t = t + 1;
end

% Final binarization
final_bin = sigmoidTransfer(Xeq1);
Sf = find(final_bin == 1);
sFeat = feat(:, Sf);

% Store results
EO.sf = Sf; 
EO.ff = sFeat; 
EO.nf = length(Sf); 
EO.c  = curve; 
EO.f  = feat; 
EO.l  = label;
end

%% Sigmoid Transfer Function
function S = sigmoidTransfer(X)
    S = 1 ./ (1 + exp(-X));
    S = S > rand(size(X));  % Binary vector
end
