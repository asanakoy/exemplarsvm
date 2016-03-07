function [ struct ] = convert_to_image_struct(I, params)
%CONVERT_TO_IMAGE_STRUCT Convert I into ImageStruct 
%if should_load_features_from_disk == 1, otherwise return an Image.
%   ImageStruct fields:
%                       id - image id,
%                       img[OPTIONAL] - image in RGB,
%                       flipval[OPTIONAL] - 1 if it is flipped, 0 - otherwise, TODO: probalby we don't need flipval
%                       feature[OPTIONAL] - feature representation of the
%                                           image.
%                       feature_flipped[OPTIONAL] - feature representation of the
%                                           flipped image.

assert(isfield(params, 'should_load_features_from_disk'), ...
    'params doesn''t have field ''should_load_features_from_disk''');

if params.should_load_features_from_disk == 1

    if isstruct(I)  
        if isfield(I, 'I')
            struct = convert_to_image_struct(I.I, params);
        elseif (~isfield(I, 'flipval') || (isfield(I, 'flipval') && I.flipval == 0)) && isfield(I, 'feature')
            struct = I;
        elseif isfield(I, 'id') && isfield(I, 'flipval')
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