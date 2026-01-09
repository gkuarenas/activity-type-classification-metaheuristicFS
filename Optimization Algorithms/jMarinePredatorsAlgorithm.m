%[2020]-"Marine Predators Algorithm: A nature-inspired metaheuristic"

% (Modified: Sigmoid-based Binarization)

function MPA = jMarinePredatorsAlgorithm(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
beta  = 1.5;   % Levy component
P     = 0.5;   % Constant
FADs  = 0.2;   % Fish aggregating devices effect

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'P'), P = opts.P; end
if isfield(opts,'FADs'), FADs = opts.FADs; end

% Objective function
fun = @jFitnessFunction;
% Number of dimensions
dim = size(feat,2); 
% Initial population
X   = rand(N,dim);  

% Preallocate
fit  = zeros(1,N);
fitG = inf;
curve = inf(1,max_Iter);
t = 1;

% Iteration
while t <= max_Iter
  % Evaluate fitness
  for i = 1:N
    S = 1 ./ (1 + exp(-X(i,:)));  % Sigmoid transfer
    bin = S >= rand(1,dim);      % Stochastic binarization
    fit(i) = fun(feat,label,bin,opts);
    
    % Global best update
    if fit(i) < fitG
      fitG = fit(i); 
      Xgb  = X(i,:);
    end
  end

  % Memory saving (first iteration)
  if t == 1
    fitM = fit; 
    Xmb  = X;
  end

  % Update memory
  for i = 1:N
    if fitM(i) < fit(i)
      fit(i) = fitM(i);
      X(i,:) = Xmb(i,:);
    end
  end
  Xmb = X;
  fitM = fit;

  % Construct elite
  Xe = repmat(Xgb, [N 1]);
  % Adaptive parameter
  CF = (1 - (t / max_Iter)) ^ (2 * (t / max_Iter));

  % --- Phase 1 ---
  if t <= max_Iter / 3
    for i = 1:N
      RB = randn(1,dim);
      for d = 1:dim
        R = rand();
        stepsize = RB(d) * (Xe(i,d) - RB(d) * X(i,d));
        X(i,d) = X(i,d) + P * R * stepsize;
      end
      X(i,:) = max(min(X(i,:),ub),lb); % Boundary check
    end

  % --- Phase 2 ---
  elseif t > max_Iter / 3 && t <= 2 * max_Iter / 3
    for i = 1:N
      if i <= N / 2
        RL = 0.05 * jLevy(beta,dim);
        for d = 1:dim
          R = rand();
          stepsize = RL(d) * (Xe(i,d) - RL(d) * X(i,d));
          X(i,d) = X(i,d) + P * R * stepsize;
        end
      else
        RB = randn(1,dim);
        for d = 1:dim
          stepsize = RB(d) * (RB(d) * Xe(i,d) - X(i,d));
          X(i,d) = Xe(i,d) + P * CF * stepsize;
        end
      end
      X(i,:) = max(min(X(i,:),ub),lb);
    end

  % --- Phase 3 ---
  else
    for i = 1:N
      RL = 0.05 * jLevy(beta,dim);
      for d = 1:dim
        stepsize = RL(d) * (RL(d) * Xe(i,d) - X(i,d));
        X(i,d) = Xe(i,d) + P * CF * stepsize;
      end
      X(i,:) = max(min(X(i,:),ub),lb);
    end
  end

  % Fitness evaluation again
  for i = 1:N
    S = 1 ./ (1 + exp(-X(i,:)));
    bin = S >= rand(1,dim);
    fit(i) = fun(feat,label,bin,opts);
    
    if fit(i) < fitG
      fitG = fit(i);
      Xgb  = X(i,:);
    end
  end

  % Memory saving again
  for i = 1:N
    if fitM(i) < fit(i)
      fit(i) = fitM(i); 
      X(i,:) = Xmb(i,:);
    end
  end
  Xmb = X;
  fitM = fit;

  % Eddy formation and FADs effect
  if rand() <= FADs
    for i = 1:N
      U = rand(1,dim) < FADs;
      R = rand();
      X(i,:) = X(i,:) + CF * (lb + R * (ub - lb)) .* U;
      X(i,:) = max(min(X(i,:),ub),lb);
    end
  else
    r = rand();
    Xr1 = X(randperm(N),:);
    Xr2 = X(randperm(N),:);
    for i = 1:N
      X(i,:) = X(i,:) + (FADs * (1 - r) + r) * (Xr1(i,:) - Xr2(i,:));
      X(i,:) = max(min(X(i,:),ub),lb);
    end
  end

  % Save convergence
  curve(t) = fitG;
  fprintf('\nIteration %d Best (MPA)= %f',t,curve(t))
  t = t + 1;
end

% Final sigmoid binarization of best position
S  = 1 ./ (1 + exp(-Xgb));
bin = S >= rand(1,dim);
Sf  = find(bin == 1); 
sFeat = feat(:,Sf);

% Store results
MPA.sf = Sf;
MPA.ff = sFeat;
MPA.nf = length(Sf);
MPA.c  = curve;
MPA.f  = feat;
MPA.l  = label;

end

% Levy distribution
function LF = jLevy(beta,dim)
num   = gamma(1 + beta) * sin(pi * beta / 2); 
deno  = gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2); 
sigma = (num / deno) ^ (1 / beta);
u     = random('Normal',0,sigma,1,dim);
v     = random('Normal',0,1,1,dim);
LF    = u ./ (abs(v) .^ (1 / beta));
end
