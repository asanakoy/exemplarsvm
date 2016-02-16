function [ struct ] = convert_to_image_struct(I, params)
%CONVERT_TO_IMAGE_STRUCT Convert I into ImageStruct if features_type is
%FeatureVector, otherwise return an Image
%   ImageStruct fields:
%                       id - image id,
%                       img - image in RGB,
%                       flipval - 1 if it is flipped, 0 - otherwise, 
%                       feature[OPTIONAL] - feature representation of the
%                                           image.

assert(isfield(params, 'features_type'), 'params doesn''t have field ''features_type''');
if strcmp(params.features_type, 'FeatureVector')

    if isstruct(I) 
        if isfield(I, 'id') && isfield(I, 'flipval')
            struct = I;
        else
            error('Unknown input image struct! We don''t know its ID or FLIPVAL')
        end
    elseif isnumeric(I) && length(size(I)) == 2 && size(I, 1) == 1 % I is a numeric vector
        struct.feature = I;
    else
        error('Unknown input image data! It''s not 1xN vector! Its dimensions: %s', mat2str(size(I)))
    end

else
    struct = convert_to_I(I);
end

end