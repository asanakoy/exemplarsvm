function [m] = esvm_train_svm_at_once(m, positives, negatives)
% Perform SVM training for a single exemplar model.
% It will be trained at once on the training set that must be loaded in m.neg_train_set 
% Some params from m.mining_params will be used.

%if no inputs are specified, just return the suffix of current method
if nargin == 0
    m = esvm_train_svm();
    return; 
end

narginchk(1, 3);

if ~exist('positives', 'var')
    assert(isfield(m , 'pos_train_set'), 'No pos_train_set field in the model!');
    % positives: [D x N_pos] matrix,
    positives = cell2mat(cellfun(@(x) x.I.feature, m.pos_train_set, 'UniformOutput', false));
    
end

if ~exist('negatives', 'var')
    assert(isfield(m , 'neg_train_set'), 'No neg_train_set field in the model!');
    % negatives: [D x N_neg] matrix
    negatives = cell2mat(cat(2, ...
                  cellfun(@(x) x.I.feature, m.neg_train_set, 'UniformOutput', false), ...
                  cellfun(@(x) x.I.feature_flipped, m.neg_train_set, 'UniformOutput', false)));
end

new_model = esvm_train_svm(positives, negatives, m.mining_params.positive_class_svm_weight, ...
    m.mining_params.train_svm_c);

m.model.svxs = new_model.svxs;
m.model.b = new_model.b;
m.model.w = reshape(new_model.w, size(m.model.w));

end
