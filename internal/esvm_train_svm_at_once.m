function [m] = esvm_train_svm_at_once(m, negatives)
% Perform SVM training for a single exemplar model.
% It will be trained at once on the training set that must be loaded in m.train_set 
% Some params from m.mining_params will be used.

%if no inputs are specified, just return the suffix of current method
if nargin == 0
    m = esvm_train_svm();
    return; 
end

if ~exist('negatives', 'var')
    assert(isfield(m , 'train_set'), 'No train_set field in the model!');
    negatives = cell2mat(cat(2, ...
                  cellfun(@(x) x.I.feature, m.train_set, 'UniformOutput', false), ...
                  cellfun(@(x) x.I.feature_flipped, m.train_set, 'UniformOutput', false)));
end

new_model = esvm_train_svm(m.model.x, negatives, m.mining_params.train_positives_constant, ...
    m.mining_params.train_svm_c);

m.model.svxs = new_model.svxs;
m.model.b = new_model.b;
m.model.w = reshape(new_model.w, size(m.model.w));

end
