function [resstruct,feat_pyramid] = esvm_detect(I, models, params)
% Localize a set of models in an image.
% function [resstruct,feat_pyramid] = esvm_detect(I, models, params)
%
% If there is a small number of models (such as in per-exemplar
% mining), then fconvblas is used for detection.  If the number is
% large, then the BLOCK feature matrix method (with a single matrix
% multiplication) is used.
%
% NOTE: These local detections can be pooled with esvm_pool_exemplars_dets.m
%
% I: Input image (or already precomputed pyramid)
% models: A cell array of models to localize inside this image
%   models{:}.model.w: Learned template
%   models{:}.model.b: Learned template's offset
% params: Localization parameters (see esvm_get_default_params.m)
%
% resstruct: Sliding window output struct with 
%   resstruct.bbs{:}: Detection boxes and pyramid locations
%   resstruct.xs{:}: Detection features
% feat_pyramid: The Feature pyramid output
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if isempty(models)
  fprintf(1,'Warning: empty models in esvm_detect\n');
  resstruct.bbs{1} = zeros(0,0);
  resstruct.xs{1} = zeros(0,0);
  feat_pyramid = [];
  return;
end

if ~iscell(models)
  models = {models};
end

if isfield(models{1},'mining_params') && ~exist('params','var')
  params = models{1}.mining_params;
elseif ~exist('params','var')
  params = esvm_get_default_params;
end

if ~isfield(params,'nnmode')
 params.nnmode = '';
end

assert(isfield(params, 'should_load_features_from_disk'));

I = convert_to_image_struct(I, params);


doflip = params.detect_add_flip;

params.detect_add_flip = 0;
[rs1, t1] = esvm_detectdriver(I, models, params);
rs1 = prune_nms(rs1, params);

if doflip == 1
  params.detect_add_flip = 1;
  [rs2, t2] = esvm_detectdriver(I, models, params);
  rs2 = prune_nms(rs2, params);
else %If there is no flip, then we are done
  resstruct = rs1;
  feat_pyramid = t1;
  return;
end

%If we got here, then the flip was turned on and we need to concatenate
%results
for exemplar_id = 1:length(rs1.bbs)
  rs1.xs{exemplar_id} = cat(2,rs1.xs{exemplar_id}, ...
                  rs2.xs{exemplar_id});


  rs1.bbs{exemplar_id} = cat(1,rs1.bbs{exemplar_id},rs2.bbs{exemplar_id});
end

resstruct = rs1;

%Concatenate normal and LR pyramids
feat_pyramid = cat(1,t1,t2);
end


function [resstruct,t] = esvm_detectdriver(I, models, ...
                                             params)
assert(isfield(params, 'features_type'), 'no field features_type');

if strcmp(params.features_type, 'FeatureVector')
    [resstruct,t] = esvm_vector_detectdriver(I, models, params);
elseif strcmp(params.features_type, 'HOG-like')
    [resstruct,t] = esvm_hog_detectdriver(I, models, params);
else
    error('Unknown params.features_type: %s', params.features_type);
end

end

function [resstruct, feature] = esvm_vector_detectdriver(I, models, params)
% Just compute score F(I)*w' - b for every exemplar.

assert(params.should_load_features_from_disk == 1);
assert(isfield(params, 'features_type') && strcmp(params.features_type, 'FeatureVector'));

feature = get_feature_from_struct(I, params);

scores = cellfun2(@(x)feature' * x.model.w - x.model.b, models);

resstruct.bbs = cell(length(models), 1);
resstruct.xs =  cell(length(models), 1);

for exemplar_id = 1:length(models)
    if (scores{exemplar_id} < params.detect_keep_threshold)
        continue
    end
    
    resstruct.bbs{exemplar_id}(:, 1:4) = [1, 1, 227, 227]; %  bounding box coordinates
    resstruct.bbs{exemplar_id}(:,5) = 1; % number of total detected bounding boxes
    resstruct.bbs{exemplar_id}(:,6) = exemplar_id;
    resstruct.bbs{exemplar_id}(:,7) = params.detect_add_flip; % flipval
    resstruct.bbs{exemplar_id}(:,8) = 1.0; % scale
    resstruct.bbs{exemplar_id}(:,9) = 1; % (bbs((i,9)), bbs(i,10)) is 2D index of convolution that gives the score
    resstruct.bbs{exemplar_id}(:,10) = 1;
    resstruct.bbs{exemplar_id}(:,11) = 0; % sample index in the mining queue. Will be filled later.
    resstruct.bbs{exemplar_id}(:,12) = scores{exemplar_id};
    
    if params.detect_save_features == 1
        resstruct.xs{exemplar_id}{1} = feature; % detection features
    end
end

end


function [resstruct, t] = esvm_hog_detectdriver(I, models, ...
                                             params)
                                     
if ~isfield(params,'max_models_before_block_method')
  params.max_models_before_block_method = 20;
end

if (length(models) > params.max_models_before_block_method) ...
      || (~isempty(params.nnmode))
    
  error('Artem has demolished this code.')
  % TODO: figure out what is it
  [resstruct,t] = esvm_detectdriverBLOCK(I, models, ...
                                         params);
  return;
end

number_of_models = length(models);
weights = cellfun2(@(x) double(x.model.w), models);
biases  = cellfun2(@(x)x.model.b, models);

%NOTE: all exemplars in this set must have the same sbin
luq = 1;

if isfield(models{1}.model,'init_params')
  sbins = cellfun(@(x)x.model.init_params.sbin,models);
  luq = length(unique(sbins));
  assert(luq == 1, 'Different bin sizes are not supported!');
end

if isfield(models{1}.model,'init_params') && luq == 1
  sbin = models{1}.model.init_params.sbin; % TODO: change sbin for conv5 features!
elseif ~isfield(models{1}.model,'init_params')
  if isfield(params,'init_params')
    sbin = params.init_params.sbin;
  else
    fprintf(1,'No hint for sbin!\n');
    error('No sbin provided');
  end
  
else
  fprintf(1,['Warning: not all exemplars have save sbin, using' ...
             ' first]\n']);
  sbin = models{1}.model.init_params.sbin;
end

t = get_pyramid(I, params);

resstruct.padder = t.padder;
resstruct.bbs = cell(number_of_models,1);
xs = cell(number_of_models,1);

maxers = cell(number_of_models,1); % maxer{i} contains maximal detection score for model i.
for q = 1:number_of_models
  maxers{q} = -inf;
end


%start with smallest level first
for level = length(t.hog):-1:1
  featr = double(t.hog{level});
  if params.dfun == 1
    featr_squared = featr.^2;
    
    %Use blas-based fast convolution code
    rootmatch1 = fconvblas(featr_squared, weights, 1, number_of_models);
    rootmatch2 = fconvblas(featr, ws2, 1, number_of_models);
     
    for z = 1:length(rootmatch1)
      rootmatch{z} = rootmatch1{z} + rootmatch2{z} + special_offset(z);
    end
  else  
    %Use blas-based fast convolution code
    assert(all(length(size(featr)) == length(size(weights{1}))), ...
        'feature vector(%s) and wight(%s) have different size!', ...
        mat2str(size(featr)), mat2str(size(weights{1})));
    rootmatch = fconvblas(featr, weights, 1, number_of_models);
  end
  
  rootmatch_sizes = cellfun2(@(x)size(x), ...
                     rootmatch);
  
  for exemplar_id = 1:number_of_models
    if prod(rootmatch_sizes{exemplar_id}) == 0
      continue
    end

    cur_scores = rootmatch{exemplar_id} - biases{exemplar_id};
    [scores,indexes] = sort(cur_scores(:),'descend');
    NKEEP = sum((scores>maxers{exemplar_id}) & (scores>=params.detect_keep_threshold));
    scores = scores(1:NKEEP);
    indexes = indexes(1:NKEEP);
    if NKEEP==0
      continue
    end
    sss = size(weights{exemplar_id});
    
    [uus,vvs] = ind2sub(rootmatch_sizes{exemplar_id}(1:2),...
                        indexes);
    
    scale = t.scales(level);
    
    o = [uus vvs] - t.padder;

    bbs = ([o(:,2) o(:,1) o(:,2)+size(weights{exemplar_id},2) ...
               o(:,1)+size(weights{exemplar_id},1)] - 1) * ...
             sbin/scale + 1 + repmat([0 0 -1 -1],length(uus),1);
    % bbs(:, 1:4) is bounding box coordinates
    bbs(:,5:12) = 0;
    bbs(:,5) = (1:size(bbs,1)); % number of total detected bounding boxes
    bbs(:,6) = exemplar_id;
    bbs(:,8) = scale;
    bbs(:,9) = uus; % (uus(i), vvs(i)) is 2D index of convolution that giebs the score scores(i)
    bbs(:,10) = vvs;
    bbs(:,12) = scores;
    
    if (params.detect_add_flip == 1)
      bbs = flip_box(bbs,t.size);
      bbs(:,7) = 1;
    end
    
    resstruct.bbs{exemplar_id} = cat(1,resstruct.bbs{exemplar_id},bbs);
    
    if params.detect_save_features == 1
      for z = 1:NKEEP
        xs{exemplar_id}{end+1} = ...
            reshape(t.hog{level}(uus(z)+(1:sss(1))-1, ...
                                 vvs(z)+(1:sss(2))-1,:), ...
                    [],1);
      end
    end
        
    if (NKEEP > 0)
      newtopk = min(params.detect_max_windows_per_exemplar,size(resstruct.bbs{exemplar_id},1));
      % Fast way to find K smallest elements in a vector. 
      % scores - is the min values, bb - their indices.
      [scores,bb] = psort(-resstruct.bbs{exemplar_id}(:,end),newtopk);
      resstruct.bbs{exemplar_id} = resstruct.bbs{exemplar_id}(bb,:);
      if params.detect_save_features == 1
        xs{exemplar_id} = xs{exemplar_id}(:,bb);
      end
      %TJM: changed so that we only maintain 'maxers' when topk
      %elements are filled
      if (newtopk >= params.detect_max_windows_per_exemplar)
        maxers{exemplar_id} = min(-scores);
      end
    end    
  end
end

if params.detect_save_features == 1
  resstruct.xs = xs;
else
  resstruct.xs = cell(number_of_models,1);
end
%fprintf(1,'\n');
end


function rs = prune_nms(rs, params)
%Prune via nms to eliminate redundant detections

%If the field is missing, or it is set to 1, then we don't need to
%process anything.  If it is zero, we also don't do NMS.
if ~isfield(params,'detect_exemplar_nms_os_threshold') || (params.detect_exemplar_nms_os_threshold >= 1) ...
      || (params.detect_exemplar_nms_os_threshold == 0)
  return;
end

rs.bbs = cellfun2(@(x)esvm_nms(x,params.detect_exemplar_nms_os_threshold),rs.bbs);

if ~isempty(rs.xs)
  for i = 1:length(rs.bbs)
    if ~isempty(rs.xs{i})
      %NOTE: the fifth field must contain elements
      rs.xs{i} = rs.xs{i}(:,rs.bbs{i}(:,5) );
    end
  end
end
end

function feature = get_feature_from_struct(I, params)
assert(isstruct(I) && isfield(I, 'id'));

if ~isfield(I, 'feature')
    assert(false, 'Features must be precomputed before!');
    % feature = params.features(I, params);
else
    if params.detect_add_flip == 0
        feature = I.feature;
    else
        feature = I.feature_flipped;
    end
end

end

function t = get_pyramid(I, params)
% Extract feature pyramid from variable I (which could be either an image,
% or already a feature pyramid)

assert(~strcmp(params.features_type, 'FeatureVector'), 'Wrong function for features_type: FeatureVector');

if isnumeric(I)
  assert(params.should_load_features_from_disk == 0);
  
  if (params.detect_add_flip == 1)
    I = flip_image(I);
  else    
    %take unadulterated "aka" un-flipped image
  end
  
  clear t
  t.size = size(I);

  %Compute pyramid
  [t.hog, t.scales] = esvm_pyramid(I, params);
  t.padder = params.detect_pyramid_padding;
  for level = 1:length(t.hog)
    t.hog{level} = padarray(t.hog{level}, [t.padder t.padder 0], 0);
  end
  
  minsizes = cellfun(@(x)min([size(x,1) size(x,2)]), t.hog);
  t.hog = t.hog(minsizes >= t.padder*2);
  t.scales = t.scales(minsizes >= t.padder*2);  
else
  fprintf(1,'Already found features\n');
  
  if iscell(I)
    if params.detect_add_flip==1
      t = I{2};
    else
      t = I{1};
    end
  elseif isstruct(I)
      assert(params.should_load_features_from_disk == 1);
      
      feature = get_feature_from_struct(I, params);
      t.hog{1} = feature;
      t.scales(1) = 1;
      t.size{1} = [227 227 3]; % TODO: Make as a parameter.
      t.padder = params.detect_pyramid_padding;
      
      t.hog{1} = padarray(t.hog{1}, [t.padder t.padder 0], 0);
  else
    assert(false, 'Warning! I dont knnow what kind of features are here!'); % Artem: may be removed ??
    t = I;
  end
end
end

