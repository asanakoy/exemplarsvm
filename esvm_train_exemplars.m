function [newmodels, new_models_name] = ...
    esvm_train_exemplars(models, train_set, params)
% Train models using train_set as negatives.
% No negative mining will be used. The model will be trained on all train_set
% at once.
% [models]: a cell array of initialized exemplar models
% [train_set]: a virtual set of negative images
% [params]: localization and training parameters

% Copyright (C) 2015-16 by Artsiom Sanakoyeu
% All rights reserved.

if isempty(models)
    fprintf('WARNING: NO INITIAL MODELS PASSED!\n');
    newmodels = models;
    new_models_name = '';
    return;
end

if isempty(params.dataset_params.localdir)
    CACHE_FILE = 0;
    fprintf('WARNING: models will not be saved on disk. Output dir is not found!\n');
else
    CACHE_FILE = 1;
end

models_name = models{1}.models_name;
new_models_name = [models_name params.training_function()];

cache_dir =  ...
    sprintf('%s/models/',params.dataset_params.localdir);

cache_file = ...
    sprintf('%s/%s.mat',cache_dir,new_models_name);

cache_file_stripped = ...
    sprintf('%s/%s-stripped.mat',cache_dir,new_models_name);

if CACHE_FILE == 1 && fileexists(cache_file_stripped)
    newmodels = load(cache_file_stripped);
    newmodels = newmodels.models;
    return;
end

if CACHE_FILE == 1 && fileexists(cache_file)
    newmodels = load(cache_file);
    newmodels = newmodels.models;
    return;
end

final_directory = ...
    sprintf('%s/models/%s/',params.dataset_params.localdir,...
    new_models_name);

%make results directory if needed
if CACHE_FILE == 1 && ~exist(final_directory,'dir')
    mkdir(final_directory);
end

result_model_handles = cell(length(models), 1); % can store either models or filepathes.
for i = 1:length(models)
    
    m = models{i};
    
    [complete_file] = sprintf('%s/%s.mat',final_directory,m.name);
    [basedir, basename, ext] = fileparts(complete_file);
    model_output_filepath = sprintf('%s/%s.mat', basedir,basename);
    
    result_model_handles{i} = model_output_filepath;
    
    % Check if we are ready for an update
    filerlock = [model_output_filepath '.mining.lock'];
    
    if CACHE_FILE == 1
        if fileexists(model_output_filepath) || (mymkdir_dist(filerlock) == 0)
            continue
        end
    end
    
    % Add training set
    m.train_set = train_set;
    
    % Add mining_params, and params.dataset_params to this exemplar
    m.mining_params = params;
    m.dataset_params = params.dataset_params;
    
    % Append '-svm' to the mode to create the models name
    m.models_name = new_models_name;
    
    m = esvm_train_svm_at_once(m);
    
    %HACK: remove train_set which causes save issue when it is a
    %cell array of function pointers
    m = rmfield(m, 'train_set');
    
    %Save the current result
    if CACHE_FILE == 1
        savem(model_output_filepath, m);
    else
        result_model_handles{i} = m;
    end
    fprintf(1,' ### End of training... \n');
    
    
    try
        if CACHE_FILE == 1
            rmdir(filerlock);
        end
    catch
        fprintf(1,'Cannot delete %s\n',filerlock);
    end
end

if CACHE_FILE == 0 % handles contain models.
    newmodels = result_model_handles;
else % handles contain flepathes of the models
    
    [result_model_handles] = sort(result_model_handles);
    %Load all of the initialized exemplars
    CACHE_FILE = 1;
    STRIP_FILE = 0; % NO STRIPPED MODELS
    
    if new_models_name(1) == '-'
        CACHE_FILE = 0;
        STRIP_FILE = 0;
    end
    
    DELETE_INITIAL = 0;
    newmodels = esvm_load_models(params.dataset_params, new_models_name, result_model_handles, ...
        CACHE_FILE, STRIP_FILE, DELETE_INITIAL);
end

end

function savem(filepath, m)
save(filepath, 'm');
end
