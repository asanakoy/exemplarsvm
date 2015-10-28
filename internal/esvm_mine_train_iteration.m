function [m] = esvm_mine_train_iteration(m, training_function)
%% ONE ITERATION OF: Mine negatives until cache is full and update the current
% classifier using training_function (do_svm, do_rank, ...). m must
% contain the field m.train_set, which indicates the current
% training set of negative images
% Returns the updated model (where m.mining_queue is updated mining_queue)
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

% Start wtrace (trace of learned classifier parameters across
% iterations) with first round classifier, if not present already
if ~isfield(m.model,'wtrace')
  m.model.wtrace{1} = m.model.w;
  m.model.btrace{1} = m.model.b;
end

if length(m.mining_queue) == 0
  fprintf(1,' ---Null mining queue, not mining!\n');
  return;
end

%If the skip is enabled, we just update the model
if m.mining_params.train_skip_mining == 0
  t_start = tic;
  [hn, m.mining_queue, mining_stats] = ...
      esvm_mine_negatives({m}, m.mining_queue, m.train_set, ...
                     m.mining_params);
  fprintf('Elapsed time for negatives mining iteration = %.2f seconds.\n', toc(t_start));
  
  if ~isfield(m.model, 'svxs') 
      num_vectors = 0;
  else
      num_vectors = size(m.model.svxs, 2);
  end
  fprintf('----Num of vectors in model before updating: %d\n', num_vectors);
  m = add_new_detections(m, cat(2,hn.xs{1}{:}), cat(1,hn.bbs{1}{: ...
                   }));
  fprintf('----Num of vectors in model after add_new_detection: %d\n', size(m.model.svxs, 2));     
else
  mining_stats.num_visited = 0;
  fprintf(1,'WARNING: train_skip_mining==1, just updating model\n');  
end
   
m = update_the_model(m, mining_stats, training_function);

if isfield(m,'dataset_params') && m.dataset_params.display == 1
  esvm_dump_figures(m);
end

function [m] = update_the_model(m, mining_stats, training_function)
%% UPDATE the current SVM, keep max number of svs, and show the results

if ~isfield(m,'mining_stats')
  m.mining_stats{1} = mining_stats;
else
  m.mining_stats{end+1} = mining_stats;
end

m = training_function(m); % = esvm_updade_svm(m)

% Append new w to trace
m.model.wtrace{end+1} = m.model.w;
m.model.btrace{end+1} = m.model.b;

% if (m.mining_params.dfun == 1)
%   r = m.model.w(:)'*bsxfun(@minus,m.model.svxs,m.model.x(:,1)).^2 - ...
%       m.model.b;
% else
%   r = m.model.w(:)'*m.model.svxs - m.model.b;
% end
% m.model.svbbs(:,end) = r;

function m = add_new_detections(m, xs, bbs)
% Add current detections (xs,bbs) to the model struct (m)
% making sure we prune away duplicates, and then sort by score
%
% Tomasz Malisiewicz (tomasz@cmu.edu)

%First iteration might not have support vector information stored
if ~isfield(m.model, 'svxs') || isempty(m.model.svxs)
  m.model.svxs = [];
  m.model.svbbs = [];
end

m.model.svxs = cat(2,m.model.svxs,xs);
m.model.svbbs = cat(1,m.model.svbbs,bbs);

%Create a unique string identifier for each of the supports
names = cell(size(m.model.svbbs,1),1);
for i = 1:length(names)
  bb = m.model.svbbs(i,:);
  names{i} = sprintf('%d.%.3f.%d.%d.%d',bb(11),bb(8), ...
                             bb(9),bb(10),bb(7));
end
  
[unames,subset,j] = unique(names);
m.model.svbbs = m.model.svbbs(subset,:);
m.model.svxs = m.model.svxs(:,subset);

[aa,bb] = sort(m.model.w(:)'*m.model.svxs,'descend');
m.model.svbbs = m.model.svbbs(bb,:);
m.model.svxs = m.model.svxs(:,bb);
