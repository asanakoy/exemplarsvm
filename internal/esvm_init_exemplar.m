function model = esvm_init_exemplar(I, bbox, init_params)
%ESVM_INIT_EXEMPLAR Initialize exemplar
%   Detailed explanation goes here

assert(isfield(init_params,'features_type'))

if strcmp(init_params.features_type, 'FeatureVector')
    model = esvm_initialize_vector_exemplar(I, bbox, init_params);
elseif strcmp(init_params.features_type, 'HOG-like')
    model = esvm_initialize_goalsize_exemplar(I, bbox, init_params);
else
    error('Unknown init_params.features_type: %s', init_params.features_type);
end

end

