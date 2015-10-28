function esvm_dump_figures(m, train_set)

if ~isfield(m, 'train_set') && nargin > 1
    m.train_set = train_set;
end
if ~isfield(m, 'train_set')
    error('Cannot show figures! train_set is not specified!\n');
end

% figure(1)
% clf
% show_cool_os(m)

% if (mining_params.dump_images == 1) || ...
%       (mining_params.dump_last_image == 1 && ...
%        m.iteration == mining_params.train_max_mine_iterations)
%   set(gcf,'PaperPosition',[0 0 10 3]);
%   print(gcf,sprintf('%s/%s.%d_iter=%05d.png', ...
%                     mining_params.final_directory,m.curid,...
%                     m.objectid,m.iteration),'-dpng'); 
% end

figure(2)
clf
Isv1 = esvm_show_det_stack(m,9);

imagesc(Isv1)
axis image
axis off
iter = length(m.model.wtrace)-1;
title(sprintf('Ex %s.%d.%s SVM-iter=%03d',m.curid,m.objectid, ...
                        m.cls(1:min(length(m.cls), 10)),iter))
drawnow
snapnow

if (m.mining_params.dump_images == 1) || ...
      (m.mining_params.dump_last_image == 1 && ...
       m.iteration == m.mining_params.train_max_mine_iterations)
  
  comment_str = '';
  if isfield(m.mining_stats{m.iteration}, 'comment')
    comment_str = ['_' m.mining_stats{m.iteration}.comment];
  end
  imwrite(Isv1,sprintf('%s/%s.%d_iter_I=%05d%s.png', ...
                    m.dataset_params.localdir, m.curid,...
                    m.objectid, m.iteration, comment_str), 'png');
end
