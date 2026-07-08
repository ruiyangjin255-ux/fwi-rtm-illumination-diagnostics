close all; clc;

% Generate shot gathers for different source types in a homogeneous
% isotropic elastic model. This wrapper keeps the source solver unchanged
% and only controls model/source/time/output parameters.

srcScript = 'D:\ryjin\elastic_fd2d\elasticFD_PML_source_fixed_data.m';
figDir = 'D:\ryjin\paper_figures_source\elastic';
dataDir = 'D:\ryjin\elastic_fd2d\saved_data';

if ~exist(figDir, 'dir')
    mkdir(figDir);
end
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

sourceIds = [1 2 3 4];
sourceTags = {'explosion', 'x_force', 'z_force', 'shear'};

for k = 1:numel(sourceIds)
    elasticFD_batch_mode = true;
    elasticFD_model_id_override = 1;       % homogeneous model
    elasticFD_source_id_override = sourceIds(k);
    elasticFD_nt_override = 1001;          % 0-1.0 s, dt = 1 ms
    elasticFD_plot_times_override = 0.8;   % keep one snapshot in data only

    elasticFD_record_out_path = fullfile(figDir, ...
        sprintf('uniform_%s_record_vx_vz.png', sourceTags{k}));
    elasticFD_data_out_path = fullfile(dataDir, ...
        sprintf('uniform_%s_record_data.mat', sourceTags{k}));

    fprintf('\nRunning homogeneous record: %s\n', sourceTags{k});
    run(srcScript);

    clear elasticFD_batch_mode elasticFD_model_id_override ...
        elasticFD_source_id_override elasticFD_nt_override ...
        elasticFD_plot_times_override elasticFD_record_out_path ...
        elasticFD_data_out_path;
end

fprintf('\nFinished multi-source homogeneous shot gathers.\n');
