%---Input-------------------------------------------------------------
% feat   : Feature vector matrix (Instances x Features)
% label  : Label matrix (Instances x 1)
% opts   : Parameter settings 
% opts.N : Number of solutions / population size (* for all methods)
% opts.T : Maximum number of iterations (* for all methods)
% opts.k : Number of k in k-nearest neighbor 

% Some methods have their specific parameters (example: PSO, GA, DE) 
% if you do not set them then they will define as default settings
% * you may open the < m.file > to view or change the parameters
% * you may use 'opts' to set the parameters of method (see example 1)
% * you may also change the < jFitnessFunction.m file >


%---Output------------------------------------------------------------
% FS    : Feature selection model (It contains several results)
% FS.sf : Index of selected features
% FS.ff : Selected features
% FS.nf : Number of selected features
% FS.c  : Convergence curve
% Acc   : Accuracy of validation model

%---Algorithms Used---------------------------------------------------
% PSO   : Particle Swarm Optimization
% GWO   : Grey Wolf Optimizer
% HHO   : Harris Hawks Optimization
% GA(t) : Genetic Algorithm (Tournament)
% WOA   : Whale Optimization Algorithm
% MPA   : Marine Predators Algorithm
% EO    : Equilibrium Optimizer
% JAYA  : JAYA Algorithm

clc; clear; close all;

%% Load dataset
data = readtable('output.csv');

%% Drop non-predictive ID/time columns
data.participant_id = [];
data.date = [];
data.health_condition = [];

%% Convert categorical features to numeric
   
categoricalVars = {'gender'};  % 'intensity' handled separately

for i = 1:length(categoricalVars)
    var = categoricalVars{i};
    if iscell(data.(var)) || iscategorical(data.(var))
        data.gender = strtrim(data.gender);                  % Trim whitespace
        emptyIdx = cellfun(@isempty, data.gender);           % Find truly empty cells
        data.gender(emptyIdx) = {''};                        % Set empty cells to empty string
        data.gender = categorical(data.gender);              % Convert to categorical
        data.gender(data.gender == "") = categorical(NaN);   % Set empty string to real NaN
        data.gender = double(data.gender);
    end
end

data.smoking_status = grp2idx(categorical(data.smoking_status));

%% Correlation Matrix
% Remove non-numeric columns
numericVars = varfun(@isnumeric, data, 'OutputFormat', 'uniform');
data_numeric = data(:, numericVars);

% Remove rows with any missing values to avoid NaNs in correlation
data_clean = rmmissing(data_numeric);

% Compute correlation matrix
corrMatrix = corr(table2array(data_clean), 'Rows', 'complete');

% Get variable names
varNames = {'Age', 'Gender', 'Height (cm)', 'Weight (kg)', 'BMI', 'Duration','Intensity', 'Calories Burned', 'Daily Steps', 'Avg Heart Rate','Resting Heart Rate', 'Systolic BP', 'Diastolic BP', 'Endurance Level', 'Sleep Hours','Stress Level', 'Hydration Level', 'Smoking Status', 'Fitness Level'};

% Display correlation heatmap
figure;
heatmap(varNames, varNames, corrMatrix, ...
    'Colormap', parula, ...
    'ColorLimits', [-1 1], ...
    'CellLabelFormat','%.2f');
set(gca, 'Fontsize', 14);
%title('Correlation Matrix of Numerical Variables');

%% Feature engineering
data.delta_heart_rate = data.avg_heart_rate - data.resting_heart_rate;
data.phys_stress_indicator = data.bmi .* data.stress_level;
data.caloric_efficiency = data.calories_burned ./ data.duration_minutes; % r > 0.5 
data.max_heart_rate = 220 -(0.7 .* data.age); %10.1016/S0735-1097(00)01054-8
data.vo2_max = 79.9 - (0.39 .* data.age) - (13.7 .* (data.gender-1)) - (0.127 .* (data.weight_kg .* 2.2)); %10.1016/j.pcad.2017.03.002

%% Define target and feature matrix
%label = grp2idx(data.activity_type);  % converted categorical to numerical
label = categorical(data.activity_type);
data.activity_type = [];

%% Get min/max values of numeric features
% Assuming your dataset is stored in a table called 'data'

% Identify numeric columns
numericVars = varfun(@isnumeric, data, 'OutputFormat', 'uniform');

% Get the names of numeric variables
numericFeatureNames = data.Properties.VariableNames(numericVars);

% Preallocate min and max arrays
minValues = zeros(sum(numericVars),1);
maxValues = zeros(sum(numericVars),1);

% Compute min and max for each numeric feature
for i = 1:length(numericFeatureNames)
    colData = data.(numericFeatureNames{i});
    minValues(i) = min(colData, [], 'omitnan');
    maxValues(i) = max(colData, [], 'omitnan');
end

% Create a table of value ranges
valueRanges = table(numericFeatureNames', minValues, maxValues, ...
    'VariableNames', {'Feature', 'MinValue', 'MaxValue'});

% Display table
disp(valueRanges);


%% Get unique class labels and their corresponding counts
[uniqueLabels, ~, idx] = unique(label);
classCounts = accumarray(idx, 1);

figure;
bar(uniqueLabels, classCounts);
xlabel('Activity Type', 'FontSize', 12);
ylabel('Count', 'FontSize', 12);
xticks(uniqueLabels);
set(gca, 'FontSize', 15); 

% Display class distribution
disp('Class distribution:');
for i = 1:length(uniqueLabels)
    fprintf('%d: %d instances\n', uniqueLabels(i), classCounts(i));
end

label = grp2idx(label);

%% Normalize Data

feat = normalize(data);

%% save train test data

writematrix(knn_xtrain, 'xtrain.csv');
writematrix(knn_ytrain, 'ytrain.csv');
writematrix(knn_xtest, 'xtest.csv');
writematrix(knn_ytest, 'ytest.csv');
%% Baseline Partition for all evaluation
HO_baseline = cvpartition(size(feat, 1), 'HoldOut', 0.2);

%% t-SNE example in MATLAB
Y = tsne(table2array(feat), 'Standardize', true);
gscatter(Y(:,1), Y(:,2), label)
title('t-SNE: Feature Separability per Activity');

%% ---- KNN Classifier (Before Feature Selection) ----

% Split features and labels
knn_xtrain = feat(HO_baseline.training, :);
knn_ytrain = label(HO_baseline.training);   
knn_xtest  = feat(HO_baseline.test, :);
knn_ytest  = label(HO_baseline.test);

% Convert tables to arrays if needed
if istable(knn_xtrain), knn_xtrain = table2array(knn_xtrain); end
if istable(knn_xtest),  knn_xtest  = table2array(knn_xtest);  end

% Convert labels to categorical if needed
if iscell(knn_ytrain), knn_ytrain = categorical(knn_ytrain); end
if iscell(knn_ytest),  knn_ytest  = categorical(knn_ytest);  end

% Train KNN classifier
K = 5;
KNNModel = fitcknn(knn_xtrain, knn_ytrain, 'NumNeighbors', K);

% Predict
knn_ypred = predict(KNNModel, knn_xtest);

% Convert to categorical (if not already)
if iscell(knn_ypred), knn_ypred = categorical(knn_ypred); end

% Confusion matrix
knn_confMat = confusionmat(knn_ytest, knn_ypred);

% Compute accuracy
knn_accuracy = sum(knn_ypred == knn_ytest) / numel(knn_ytest);

% Compute precision, recall, F1 for each class
knn_classes = unique(knn_ytest);
knn_numClasses = numel(knn_classes);

knn_precision = zeros(knn_numClasses, 1);
knn_recall    = zeros(knn_numClasses, 1);
knn_f1        = zeros(knn_numClasses, 1);

for i = 1:knn_numClasses
    knn_TP = knn_confMat(i, i);
    knn_FP = sum(knn_confMat(:, i)) - knn_TP;
    knn_FN = sum(knn_confMat(i, :)) - knn_TP;

    knn_precision(i) = knn_TP / (knn_TP + knn_FP + eps);
    knn_recall(i)    = knn_TP / (knn_TP + knn_FN + eps);
    knn_f1(i)        = 2 * (knn_precision(i) * knn_recall(i)) / (knn_precision(i) + knn_recall(i) + eps);
end

% Display results
fprintf('\n--- KNN Classifier (Before Feature Selection) ---\n');
fprintf('Overall Accuracy: %.6f\n', knn_accuracy);

for i = 1:knn_numClasses
    fprintf('Class: %g\n', knn_classes(i));
    fprintf('  Precision: %.6f\n', knn_precision(i));
    fprintf('  Recall   : %.6f\n', knn_recall(i));
    fprintf('  F1 Score : %.6f\n', knn_f1(i));
end

%% Metaheuristics for Feature selection
% Parameters
runs = 20;
opts.k=5;

%% 1. Particle Swarm Optimization
opts.N     = 30;      % Number of particles
opts.T     = 100;     % Max iterations
opts.c1    = 2;
opts.c2    = 2;
opts.Model = HO_baseline;
PSO_FS = cell(1,runs);
PSO_fitness = zeros(1,runs);

% Run PSO 30 times

fprintf('\n======= PSO for Feature Selection ======');

for i = 1:runs
    fprintf('\nRunning PSO iteration %d/%d...\n', i, runs);
    FS = jfs('pso', feat, label, opts);  % Call your BPSO function
    PSO_FS{i} = FS;
    PSO_fitness(i) = FS.c(end);  % Final fitness (best of that run)
end

% Find best run
[~, best_idx] = min(PSO_fitness);  % lower fitness is better
PSO_best_FS = PSO_FS{best_idx};        % best feature selection result
PSO_sf_idx = PSO_best_FS.sf;               % selected feature indices

% Evaluate accuracy using selected features
PSO_Acc = jknn(feat(:,PSO_sf_idx), label, opts);  % Final KNN classification accuracy

% Plot convergence curve of best run
plot(PSO_best_FS.c);
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('Best PSO Run Convergence');
grid on;

% Optional: Output number of selected features and accuracy
fprintf('\nBest Run: Selected %d features\n', length(PSO_sf_idx));
fprintf('Classification Accuracy: %.6f\n', PSO_Acc);

%% 2. Grey Wolf Optimizer
opts.N     = 30;      % Number of wolves
opts.T     = 100;     % Max iterations
opts.Model = HO_baseline;
GWO_FS_all = cell(1,runs);
GWO_fitness = zeros(1,runs);

% Run PSO 30 times

fprintf('======= GWO for Feature Selection ======');

for i = 1:runs
    fprintf('\nRunning GWO iteration %d/%d...\n', i, runs);
    GWO_FS = jfs('gwo', feat, label, opts);  % Call your BGWO function
    GWO_FS_all{i} = GWO_FS;
    GWO_fitness(i) = GWO_FS.c(end);  % Final fitness (best of that run)
end

% Find best run
[~, best_idx] = min(GWO_fitness);  % lower fitness is better
GWO_best_FS = GWO_FS_all{best_idx};         % best feature selection result
GWO_sf_idx = GWO_best_FS.sf;               % selected feature indices

% Evaluate accuracy using selected features
GWO_Acc = jknn(feat(:,GWO_sf_idx), label, opts);  % Final KNN classification accuracy

% Plot convergence curve of best run
plot(GWO_best_FS.c);
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('Best GWO Run Convergence');
grid on;

% Optional: Output number of selected features and accuracy
fprintf('\nBest Run: Selected %d features\n', length(GWO_sf_idx));
fprintf('Classification Accuracy: %.6f\n', GWO_Acc);

%% 3. Harris Hawks Optimization

opts.N     = 30;      % Number of wolves
opts.T     = 100;     % Max iterations
opts.Model = HO_baseline;
HHO_FS_all = cell(1,runs);
HHO_fitness = zeros(1,runs);

% Run HHO 30 times

fprintf('======= HHO for Feature Selection ======');

for i = 1:runs
    fprintf('\nRunning HHO iteration %d/%d...\n', i, runs);
    HHO_FS = jfs('hho', feat, label, opts);  % Call your BHHO function
    HHO_FS_all{i} = HHO_FS;
    HHO_fitness(i) = HHO_FS.c(end);  % Final fitness (best of that run)
end

% Find best run
[~, best_idx] = min(HHO_fitness);  % lower fitness is better
HHO_best_FS = HHO_FS_all{best_idx};         % best feature selection result
HHO_sf_idx = HHO_best_FS.sf;               % selected feature indices

% Evaluate accuracy using selected features
HHO_Acc = jknn(feat(:,HHO_sf_idx), label, opts);  % Final KNN classification accuracy

% Plot convergence curve of best run
plot(HHO_best_FS.c);
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('Best HHO Run Convergence');
grid on;

% Optional: Output number of selected features and accuracy
fprintf('\nBest Run: Selected %d features\n', length(HHO_sf_idx));
fprintf('Classification Accuracy: %.6f\n', HHO_Acc);

%% 4. Genetic Algorithm

opts.N     = 30;      % Number of genes
opts.T     = 100;     % Max iterations
opts.Model = HO_baseline;
GA_FS_all = cell(1,runs);
GA_fitness = zeros(1,runs);

% Run HHO 20 times

fprintf('======= GA for Feature Selection ======');

for i = 1:runs
    fprintf('\nRunning GA iteration %d/%d...\n', i, runs);
    GA_FS = jfs('gat', feat, label, opts);  % Call your BHHO function
    GA_FS_all{i} = GA_FS;
    GA_fitness(i) = GA_FS.c(end);  % Final fitness (best of that run)
end

% Find best run
[~, best_idx] = min(GA_fitness);  % lower fitness is better
GA_best_FS = GA_FS_all{best_idx};         % best feature selection result
GA_sf_idx = GA_best_FS.sf;               % selected feature indices

% Evaluate accuracy using selected features
GA_Acc = jknn(feat(:,GA_sf_idx), label, opts);  % Final KNN classification accuracy

% Plot convergence curve of best run
plot(GA_best_FS.c);
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('Best GA Run Convergence');
grid on;

% Optional: Output number of selected features and accuracy
fprintf('\nBest Run: Selected %d features\n', length(GA_sf_idx));
fprintf('Classification Accuracy: %.6f\n', GA_Acc);

%% 5.Equilibrium Optimizer

opts.N     = 30;      % Number of bears
opts.T     = 100;     % Max iterations
opts.Model = HO_baseline;
EO_FS_all = cell(1,runs);
EO_fitness = zeros(1,runs);

% Run HHO 20 times

fprintf('======= EO for Feature Selection ======');

for i = 1:runs
    fprintf('\nRunning EO iteration %d/%d...\n', i, runs);
    EO_FS = jfs('eo', feat, label, opts);  % Call your EO function
    EO_FS_all{i} = EO_FS;
    EO_fitness(i) = EO_FS.c(end);  % Final fitness (best of that run)
end

% Find best run
[~, best_idx] = min(EO_fitness);  % lower fitness is better
EO_best_FS = EO_FS_all{best_idx};         % best feature selection result
EO_sf_idx = EO_best_FS.sf;               % selected feature indices

% Evaluate accuracy using selected features
EO_Acc = jknn(feat(:,EO_sf_idx), label, opts);  % Final KNN classification accuracy

% Plot convergence curve of best run
plot(EO_best_FS.c);
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('Best EO Run Convergence');
grid on;

% Optional: Output number of selected features and accuracy
fprintf('\nBest Run: Selected %d features\n', length(EO_sf_idx));
fprintf('Classification Accuracy: %.6f\n', EO_Acc);
%% 6. Whale Optimization Algorithm

opts.N     = 30;      % Number of bears
opts.T     = 100;     % Max iterations
opts.Model = HO_baseline;
WOA_FS_all = cell(1,runs);
WOA_fitness = zeros(1,runs);

% Run WOA 20 times

fprintf('======= WOA for Feature Selection ======');

for i = 1:runs
    fprintf('\nRunning WOA iteration %d/%d...\n', i, runs);
    WOA_FS = jfs('woa', feat, label, opts);  % Call your WOA function
    WOA_FS_all{i} = WOA_FS;
    WOA_fitness(i) = WOA_FS.c(end);  % Final fitness (best of that run)
end

% Find best run
[~, best_idx] = min(WOA_fitness);  % lower fitness is better
WOA_best_FS = WOA_FS_all{best_idx};         % best feature selection result
WOA_sf_idx = WOA_best_FS.sf;               % selected feature indices

% Evaluate accuracy using selected features
WOA_Acc = jknn(feat(:,WOA_sf_idx), label, opts);  % Final KNN classification accuracy

% Plot convergence curve of best run
plot(WOA_best_FS.c);
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('Best WOA Run Convergence');
grid on;

% Optional: Output number of selected features and accuracy
fprintf('\nBest Run: Selected %d features\n', length(WOA_sf_idx));
fprintf('Classification Accuracy: %.6f\n', WOA_Acc);

%% 7. Marine Predators Algorithm
opts.N     = 30;      % Number of bears
opts.T     = 100;     % Max iterations
opts.Model = HO_baseline;
MPA_FS_all = cell(1,runs);
MPA_fitness = zeros(1,runs);

% Run MPA 20 times

fprintf('======= MPA for Feature Selection ======');

for i = 1:runs
    fprintf('\nRunning MPA iteration %d/%d...\n', i, runs);
    MPA_FS = jfs('mpa', feat, label, opts);  % Call your MPA function
    MPA_FS_all{i} = MPA_FS;
    MPA_fitness(i) = MPA_FS.c(end);  % Final fitness (best of that run)
end

% Find best run
[~, best_idx] = min(MPA_fitness);  % lower fitness is better
MPA_best_FS = MPA_FS_all{best_idx};         % best feature selection result
MPA_sf_idx = MPA_best_FS.sf;               % selected feature indices

% Evaluate accuracy using selected features
MPA_Acc = jknn(feat(:,MPA_sf_idx), label, opts);  % Final KNN classification accuracy

% Optional: Output number of selected features and accuracy
fprintf('\nBest Run: Selected %d features\n', length(MPA_sf_idx));
fprintf('Classification Accuracy: %.6f\n', MPA_Acc);

%% 8. Jaya Algorithm
opts.N     = 30;      % Number of bears
opts.T     = 100;     % Max iterations
opts.Model = HO_baseline;
JA_FS_all = cell(1,runs);
JA_fitness = zeros(1,runs);

% Run JA 20 times

fprintf('======= JA for Feature Selection ======');

for i = 1:runs
    fprintf('\nRunning JA iteration %d/%d...\n', i, runs);
    JA_FS = jfs('ja', feat, label, opts);  % Call your JA function
    JA_FS_all{i} = JA_FS;
    JA_fitness(i) = JA_FS.c(end);  % Final fitness (best of that run)
end

% Find best run
[~, best_idx] = min(JA_fitness);  % lower fitness is better
JA_best_FS = JA_FS_all{best_idx};         % best feature selection result
JA_sf_idx = JA_best_FS.sf;               % selected feature indices

% Evaluate accuracy using selected features
JA_Acc = jknn(feat(:,JA_sf_idx), label, opts);  % Final KNN classification accuracy

% Optional: Output number of selected features and accuracy
fprintf('\nBest Run: Selected %d features\n', length(JA_sf_idx));
fprintf('Classification Accuracy: %.6f\n', JA_Acc);

%% Display Convergence Curves

figure;
% Plot all results in a convergence plot
plot(PSO_best_FS.c, 'Linewidth', 2); hold on;
plot(GWO_best_FS.c, 'Linewidth', 2);
plot(HHO_best_FS.c, 'Linewidth', 2);
plot(GA_best_FS.c, 'Linewidth', 2);
plot(EO_best_FS.c, 'Linewidth', 2);
plot(WOA_best_FS.c, 'Linewidth', 2);
plot(MPA_best_FS.c, 'Linewidth', 2);
plot(JA_best_FS.c, 'LineWidth',2);
set(gca, 'Fontsize', 30)

hold off;
grid on;

legend('bPSO', 'bGWO', 'bHHO','bGA','bEO','bWOA', 'bMPA', 'bJA', 'Fontsize', 20);
xlabel('Iteration');
ylabel('Fitness');

%% Evaluation metrics

% Average Fitness
PSO_avgfit = mean(PSO_fitness);
GWO_avgfit = mean(GWO_fitness);
HHO_avgfit = mean(HHO_fitness);
GA_avgfit = mean(GA_fitness);
EO_avgfit = mean(EO_fitness);
WOA_avgfit = mean(WOA_fitness);
MPA_avgfit = mean(MPA_fitness);
JA_avgfit = mean(JA_fitness);

% Average Selection Size
PSO_avgselsize = mean(cellfun(@(x) x.nf, PSO_FS));
GWO_avgselsize = mean(cellfun(@(x) x.nf, GWO_FS_all));
HHO_avgselsize = mean(cellfun(@(x) x.nf, HHO_FS_all));
GA_avgselsize = mean(cellfun(@(x) x.nf, GA_FS_all));
EO_avgselsize = mean(cellfun(@(x) x.nf, EO_FS_all));
WOA_avgselsize = mean(cellfun(@(x) x.nf, WOA_FS_all));
MPA_avgselsize = mean(cellfun(@(x) x.nf, MPA_FS_all));
JA_avgselsize = mean(cellfun(@(x) x.nf, JA_FS_all));

% Best Fitness
PSO_bestfit = min(PSO_fitness);
GWO_bestfit = min(GWO_fitness);
HHO_bestfit = min(HHO_fitness);
GA_bestfit = min(GA_fitness);
EO_bestfit = min(EO_fitness);
WOA_bestfit = min(WOA_fitness);
MPA_bestfit = min(MPA_fitness);
JA_bestfit = min(JA_fitness);

% Worst Fitness
PSO_worstfit = max(PSO_fitness);
GWO_worstfit = max(GWO_fitness);
HHO_worstfit = max(HHO_fitness);
GA_worstfit = max(GA_fitness);
EO_worstfit = max(EO_fitness);
WOWOAA_worstfit = max(WOA_fitness);
MPA_worstfit = max(MPA_fitness);
JA_worstfit = max(JA_fitness);

% Stddev fitness
PSO_stdfit = std(PSO_fitness);
GWO_stdfit = std(GWO_fitness);
HHO_stdfit = std(HHO_fitness);
GA_stdfit = std(GA_fitness);
EO_stdfit = std(EO_fitness);
WOA_stdfit = std(WOA_fitness);
MPA_stdfit = std(MPA_fitness);
JA_stdfit = std(JA_fitness);

% set rows
metrics = {'Average Fitness'; 'Average Selection Size'; 'Best Fitness'; 'Worst Fitness'; 'Standard Deviation Fitness'};
PSO = [PSO_avgfit; PSO_avgselsize; PSO_bestfit; PSO_worstfit; PSO_stdfit];
GWO = [GWO_avgfit; GWO_avgselsize; GWO_bestfit; GWO_worstfit; GWO_stdfit];
HHO = [HHO_avgfit; HHO_avgselsize; HHO_bestfit; HHO_worstfit; HHO_stdfit];
GA = [GA_avgfit; GA_avgselsize; GA_bestfit; GA_worstfit; GA_stdfit];
EO = [EO_avgfit; EO_avgselsize; EO_bestfit; EO_worstfit; EO_stdfit];
WOA = [WOA_avgfit; WOA_avgselsize; WOA_bestfit; WOA_worstfit; WOA_stdfit];
MPA = [MPA_avgfit; MPA_avgselsize; MPA_bestfit; MPA_worstfit; MPA_stdfit];
JA = [JA_avgfit; JA_avgselsize; JA_bestfit; JA_worstfit; JA_stdfit];

% create table
perf_metrics_fs = table(metrics, PSO, GWO, HHO, GA, EO, WOA, MPA, JA);
writetable(perf_metrics_fs, 'A_perfmetrics_fs.csv');

%% Create SpiderPlot
% List of algorithms
algorithms = {'bPSO', 'bGWO', 'bHHO', 'bGA', 'bEO', 'bWOA', 'bMPA', 'bJA'};

% Metrics: rows = metrics, columns = algorithms
metrics = [
    PSO_avgfit, GWO_avgfit, HHO_avgfit, GA_avgfit, EO_avgfit, WOA_avgfit, MPA_avgfit, JA_avgfit;
    PSO_avgselsize, GWO_avgselsize, HHO_avgselsize, GA_avgselsize, EO_avgselsize, WOA_avgselsize, MPA_avgselsize, JA_avgselsize;
    PSO_bestfit, GWO_bestfit, HHO_bestfit, GA_bestfit, EO_bestfit, WOA_bestfit, MPA_bestfit, JA_bestfit;
    PSO_worstfit, GWO_worstfit, HHO_worstfit, GA_worstfit, EO_worstfit, WOWOAA_worstfit, MPA_worstfit, JA_worstfit;
    PSO_stdfit, GWO_stdfit, HHO_stdfit, GA_stdfit, EO_stdfit, WOA_stdfit, MPA_stdfit, JA_stdfit
];

% Normalize the data for radar plot
metrics_norm = normalize(metrics, 2, 'range');

% Transpose for plotting (now rows = algorithms, columns = metrics)
metrics_norm_t = metrics_norm';

% Metric labels
metric_labels = {'Avg Fitness', 'Avg Sel. Size', 'Best Fitness', 'Worst Fitness', 'StdDev Fitness'};

% Call the spider_plot function with minimal arguments
figure;
spider1 = spider_plot_class(metrics_norm_t, ...
    'AxesLabels', metric_labels, ...
    'AxesDisplay', 'one',...
    'AxesInterval', 4,...
    'LegendLabels', algorithms, ...
    'AxesFontSize', 14, ...
    'LabelFontSize', 14,...
    'FillOption', 'on',...
    'FillTransparency', 0.15, ...
    'AxesWebType', 'circular');

%% ---- KNN Classifier (After Feature Selection) ----

% Split features and labels
knn_xtrain_fs = feat(HO_baseline.training, GA_sf_idx);
knn_ytrain_fs = label(HO_baseline.training);   
knn_xtest_fs  = feat(HO_baseline.test, GA_sf_idx);
knn_ytest_fs  = label(HO_baseline.test);

% Convert tables to arrays if needed
if istable(knn_xtrain_fs), knn_xtrain_fs = table2array(knn_xtrain_fs); end
if istable(knn_xtest_fs),  knn_xtest_fs  = table2array(knn_xtest_fs);  end

% Convert labels to categorical if needed
if iscell(knn_ytrain), knn_ytrain = categorical(knn_ytrain); end
if iscell(knn_ytest),  knn_ytest  = categorical(knn_ytest);  end

% Train KNN classifier
K = 5;
KNNModel_fs = fitcknn(knn_xtrain_fs, knn_ytrain_fs, 'NumNeighbors', K);

% Predict
knn_ypred_fs = predict(KNNModel_fs, knn_xtest_fs);

% Convert to categorical (if not already)
if iscell(knn_ypred_fs), knn_ypred_fs = categorical(knn_ypred_fs); end

% Confusion matrix
knn_confMat_fs = confusionmat(knn_ytest_fs, knn_ypred_fs);

% Compute accuracy
knn_accuracy_fs = sum(knn_ypred_fs == knn_ytest_fs) / numel(knn_ytest_fs);

% Compute precision, recall, F1 for each class
knn_classes_fs = unique(knn_ytest_fs);
knn_numClasses_fs = numel(knn_classes_fs);

knn_precision_fs = zeros(knn_numClasses_fs, 1);
knn_recall_fs    = zeros(knn_numClasses_fs, 1);
knn_f1_fs        = zeros(knn_numClasses_fs, 1);

for i = 1:knn_numClasses_fs
    knn_TP_fs = knn_confMat_fs(i, i);
    knn_FP_fs = sum(knn_confMat_fs(:, i)) - knn_TP_fs;
    knn_FN_fs = sum(knn_confMat_fs(i, :)) - knn_TP_fs;

    knn_precision_fs(i) = knn_TP_fs / (knn_TP_fs + knn_FP_fs + eps);
    knn_recall_fs(i)    = knn_TP_fs / (knn_TP_fs + knn_FN_fs + eps);
    knn_f1_fs(i)        = 2 * (knn_precision_fs(i) * knn_recall_fs(i)) / (knn_precision_fs(i) + knn_recall_fs(i) + eps);
end

% Display results
fprintf('\n--- KNN Classifier (After Feature Selection) ---\n');
fprintf('Overall Accuracy: %.6f\n', knn_accuracy_fs);
fprintf('Overall Precision: %.6f\n', mean(knn_precision_fs));
fprintf('Overall Recall: %.6f\n', mean(knn_recall_fs));
fprintf('Overall F1-Score: %.6f\n', mean(knn_f1_fs));

for i = 1:knn_numClasses
    fprintf('Class: %g\n', knn_classes_fs(i));
    fprintf('  Precision: %.6f\n', knn_precision_fs(i));
    fprintf('  Recall   : %.6f\n', knn_recall_fs(i));
    fprintf('  F1 Score : %.6f\n', knn_f1_fs(i));
end


