function [m] = esvm_train_svm_at_once(m)
% Perform SVM training for a single exemplar model.
% It will be trained at once on the training set that must be loaded in m.train_set 
% Some params from m.mining_params will be used.

%if no inputs are specified, just return the suffix of current method
if nargin == 0
    m = esvm_train_svm();
    return; 
end

new_model = esvm_train_svm(m.x, m.train_set, m.mining_params.train_positives_constant);

m.model.svxs = new_model.svxs;
m.model.b = new_model.b;
m.model.w = reshape(new_model.w, size(m.model.w));

end

