function [model] = esvm_train_svm(positives, negatives, weight_positive, svm_c_constant)
% Perform SVM training.
% Arguments:
%           positives: [D x N_pos] matrix,
%           negatives: [D x N_neg] matrix,
%               where N is the number of
%               samples, D - num of feature vector dimensions.
%           weight_positive: % weight of the positive class
%
% Copyright (C) 2015-16 by Artsiom Sanakoyeu
% All rights reserved.

%if no inputs are specified, just return the suffix of current method
if nargin == 0
    model = '-svm';
    return;
end

superx = cat(2, positives, negatives);
supery = cat(1, ones(size(positives, 2), 1), -1 * ones(size(negatives, 2), 1) );

number_positives = sum(supery == 1);
number_negatives = sum(supery == -1);

assert(number_positives == size(positives, 2));
assert(number_negatives == size(negatives, 2));

%% Training
fprintf(1,' -----\nStarting SVM: dim=%d... #pos=%d, #neg=%d ',...
    size(superx, 1), number_positives, number_negatives);
starttime = tic;

weight_negative = 1;
svm_model = libsvmtrain(supery, superx',sprintf(['-s 0 -t 0 -c' ...
    ' %f -w1 %.9f -q'], svm_c_constant, weight_positive));

fprintf('\nSupport vectors in trained model: %d\n', length(svm_model.sv_coef));
assert(~isempty(svm_model.sv_coef), 'Empty model!')
%% ==========================================================================

%convert support vectors to decision boundary
% svm_model.SVs is [K x D]
% svm_model.sv_coef is [K x 1]
svm_weights = (svm_model.sv_coef' * full(svm_model.SVs)); % [1 x K] * [K x D] = [1 x D]

wex = svm_weights'; % [D x 1]
b = svm_model.rho;
if supery(1) == -1
    wex = wex * -1;
    b = b * -1;
end

%% issue a warning if the norm is very small
if norm(wex) < .00001
    fprintf(1,'WARNING: learning broke down!\n');
end

maxpos = max(wex' * positives - b);
fprintf(1,' --- Max positive is %.3f\n',maxpos);
fprintf(1,'SVM iteration took %.3f sec, ',toc(starttime));

model.w = wex;
model.b = b;
model.svxs = full(svm_model.SVs)'; % [D x K]. Can contain positive vectors (samples). 

r = model.w(:)' * model.svxs - model.b;
BORDER_SV_THRESHOLD = -1.0005;
num_of_sv_thresholded = sum(r >= BORDER_SV_THRESHOLD);

fprintf('Number of SV (score >= %f): %d\n', BORDER_SV_THRESHOLD, num_of_sv_thresholded);
assert(num_of_sv_thresholded > 0, 'ERROR: number of support vectors (score >= %f) is 0!\n', BORDER_SV_THRESHOLD);

end
