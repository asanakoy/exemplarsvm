function [m] = esvm_update_svm(m)
% Perform SVM learning for a single exemplar model, we assume that
% the exemplar has a set of detections loaded in m.model.svxs and m.model.svbbs
% Durning Learning, we can apply some pre-processing such as PCA or
% dominant gradient projection
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

%if no inputs are specified, just return the suffix of current method
if nargin==0
  m = '-svm';
  return;
end

if ~isfield(m.model,'mask') || isempty(m.model.mask)
  m.model.mask = true(numel(m.model.w), 1);
end

if length(m.model.mask(:)) ~= numel(m.model.w)
  m.model.mask = repmat(m.model.mask,[1 1 m.model.hg_size(3)]);
  m.model.mask = logical(m.model.mask(:));
end

mining_params = m.mining_params;
xs = m.model.svxs; % support vectors
bbs = m.model.svbbs; % support vectors bounding boxes


%NOTE: MAXSIZE should perhaps be inside of the default_params script?
MAXSIZE = 4000;
if size(xs,2) >= MAXSIZE
  fprintf('num of training images() >= MAXSIZE(%d); ', size(xs,2), MAXSIZE);
  HALFSIZE = MAXSIZE/2;
  %NOTE: random is better than top 5000
  r_neg = m.model.w(:)'*xs;
  [tmp,r_neg] = sort(r_neg,'descend');
  r1 = r_neg(1:HALFSIZE);
  
  r_neg = HALFSIZE+randperm(length(r_neg((HALFSIZE+1):end)));
  r_neg = r_neg(1:HALFSIZE);
  r_neg = [r1 r_neg];
  fprintf('num of training images kept: %d ', length(r_neg));
  xs = xs(:,r_neg);
  bbs = bbs(r_neg,:);
end

if isfield(m , 'pos_train_set')
    assert(iscell(m.pos_train_set), 'Unknown type of m.pos_train_set!');
    positives_xs = cell2mat(cellfun(@(x) x.I.feature, m.pos_train_set, 'UniformOutput', false));
else
    positives_xs = m.model.x;
end
  
superx = cat(2, positives_xs, xs);
supery = cat(1, ones(size(positives_xs, 2), 1), -1 * ones(size(xs, 2), 1) );

number_positives = sum(supery == 1);
number_negatives = sum(supery == -1);

if mining_params.auto_weight_svm_classes == 0
  wpos = mining_params.positive_class_svm_weight; % weight of the positive class
  wneg = 1.0;
else
  fprintf(1, 'automatically balancing svm classes\n');
  total_train_samples = number_positives + number_negatives;
  wpos = 1.0 * number_negatives / total_train_samples;
  wneg = 1.0 * number_positives / total_train_samples;
end

A = eye(size(superx, 1));
mu = zeros(size(superx, 1), 1);

% if mining_params.DOMINANT_GRADIENT_PROJECTION == 1  
%   A = get_dominant_basis(reshape(mean(m.model.x(:,1),2), ...
%                                  m.model.hg_size),...
%                          mining_params.DOMINANT_GRADIENT_PROJECTION_K);
  
  
%   A2 = get_dominant_basis(reshape(mean(superx(:,supery==-1),2), ...
%                                   m.model.hg_size),...
%                           mining_params ...
%                           .DOMINANT_GRADIENT_PROJECTION_K);
%   A = [A A2];
% elseif mining_params.DO_PCA == 1
%   [A,d,mu] = mypca(superx,mining_params.PCA_K);
% elseif mining_params.A_FROM_POSITIVES == 1
%   A = [superx(:,supery==1)];
%   cursize = size(A,2);
%   for qqq = 1:cursize
%     A(:,qqq) = A(:,qqq) - mean(A(:,qqq));
%     A(:,qqq) = A(:,qqq)./ norm(A(:,qqq));
%   end
  
%   %% add some ones
%   A(:,end+1) = 1;
%   A(:,end) = A(:,end) / norm(A(:,end));
% end

newx = bsxfun(@minus,superx,mu);
newx = newx(logical(m.model.mask),:);
newx = A(m.model.mask,:)'*newx;

fprintf(1,' -----\nStarting SVM: dim=%d... #pos=%d, #neg=%d ',...
        size(newx, 1), number_positives, number_negatives);
starttime = tic;

svm_model = libsvmtrain(supery, newx',sprintf(['-s 0 -t 0 -c' ...
                    ' %f -w-1 %.9f -w1 %.9f -q'], mining_params.train_svm_c, wneg, wpos));

fprintf('\nSupport vectors in trained model: %d\n', length(svm_model.sv_coef));

if isempty(svm_model.sv_coef)
  %learning had no negatives
  wex = m.model.w;
  b = m.model.b;
  fprintf(1,'reverting to old model...\n');
else
  
  % svm_model.SVs is [K x D]
  % svm_model.sv_coef is [K x 1]
  svm_weights = (svm_model.sv_coef' * full(svm_model.SVs)); % [1 x K] * [K x D] = [1 x D]
  
  wex = svm_weights'; % [D x 1]
  b = svm_model.rho;
  
  if supery(1) == -1
    wex = wex*-1;
    b = b*-1;    
  end
  
  %% project back to original space
  b = b + wex'*A(m.model.mask,:)'*mu(m.model.mask);
  wex = A(m.model.mask,:)*wex;
  
  wex2 = zeros(size(superx,1),1);
  wex2(m.model.mask) = wex;
  
  wex = wex2;
  
  %% issue a warning if the norm is very small
  if norm(wex) < .00001
    fprintf(1,'learning broke down!\n');
  end  
end

maxpos = max(wex'*m.model.x - b);
fprintf(1,' --- Max positive is %.3f\n',maxpos);
fprintf(1,'SVM iteration took %.3f sec, ',toc(starttime));

m.model.w = reshape(wex, size(m.model.w));
m.model.b = b;



r_neg = m.model.w(:)'*m.model.svxs - m.model.b;
r_libsvm = m.model.w(:)'*svm_model.SVs' - m.model.b;

BORDER_SV_THRESHOLD = -1.0005;
num_of_sv = sum((r_neg >= BORDER_SV_THRESHOLD) .* (r_neg <= -BORDER_SV_THRESHOLD)) + number_positives; % add positive vector, i.e. exemplar m.model.x and other positives(if exist)
num_of_sv_libsvm = length(svm_model.sv_coef < 0);

if (num_of_sv < num_of_sv_libsvm)
    fprintf('WARNING: num of SV (%d) < then num of SV obtained by libSVM (%d)!\n', num_of_sv, num_of_sv_libsvm);
end

fprintf('Number of SV (score in [%f, %f]): %d\n', BORDER_SV_THRESHOLD, -BORDER_SV_THRESHOLD, num_of_sv);
% fprintf('Number of SV (score >= mining_params.detect_keep_threshold(=%f)): %d\n', ...
%     mining_params.detect_keep_threshold, sum(r >= mining_params.detect_keep_threshold));

if num_of_sv == 0
  fprintf(1,' ERROR: number of negative support vectors is 0!\');
  error('Something went wrong');
end


%KEEP (nsv_multiplier * #SV) vectors, but at most max_negatives of them
max_number_of_vectors_to_keep = min(ceil(mining_params.train_keep_nsv_multiplier * num_of_sv), ...
                                    mining_params.train_max_negatives_in_cache);

[alpha, v_indices] = sort(r_neg,'descend');
v_indices = v_indices(1:min(length(v_indices),max_number_of_vectors_to_keep));
m.model.svxs = m.model.svxs(:,v_indices);
m.model.svbbs = m.model.svbbs(v_indices,:);
fprintf(1,' kept %d negatives\n',length(v_indices));
