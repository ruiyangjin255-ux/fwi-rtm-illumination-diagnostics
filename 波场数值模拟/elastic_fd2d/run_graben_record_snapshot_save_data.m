close all; clc;

srcScript = 'D:\ryjin\elastic_fd2d\elasticFD_PML_source_fixed_data.m';
dataDir = 'D:\ryjin\elastic_fd2d\saved_data';
figDir = 'D:\ryjin\paper_figures_source\elastic';

if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

elasticFD_batch_mode = true;
elasticFD_model_id_override = 3;
elasticFD_source_id_override = 1;
elasticFD_nt_override = 4001;
elasticFD_plot_times_override = [1.0 1.2 1.5];
elasticFD_data_out_path = fullfile(dataDir, 'graben_explosion_record_snapshots_4s_data.mat');
elasticFD_record_out_path = fullfile(figDir, 'graben_record_vx_vz_4s.png');

run(srcScript);

clear elasticFD_batch_mode elasticFD_model_id_override ...
    elasticFD_source_id_override elasticFD_nt_override ...
    elasticFD_plot_times_override elasticFD_data_out_path ...
    elasticFD_record_out_path;

fprintf('Saved graben elastic data and record figure.\n');
