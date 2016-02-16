function model = esvm_initialize_vector_exemplar( I, bbox, init_params )
%ESVM_INITIALIZE_VECTOR_EXEMPLAR Init exemplar svm with the
%positive sample's feature (with substracted mean).

assert(isstruct(I));

if ~isfield(I, 'feature')
    I.feature = init_params.features(I);
end

model.w = I.feature - mean(I.feature);
model.b = 0;
model.x = I.feature;
model.bb = bbox;

end

