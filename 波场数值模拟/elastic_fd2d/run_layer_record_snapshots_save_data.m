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
elasticFD_model_id_override = 2;
elasticFD_source_id_override = 1;
elasticFD_nt_override = 2001;
elasticFD_plot_times_override = [0.6 0.8 1.0 1.2];
elasticFD_data_out_path = fullfile(dataDir, 'layer_explosion_record_snapshots_data.mat');
elasticFD_record_out_path = fullfile(figDir, 'layer_record_vx_vz_2s.png');

run(srcScript);

clear elasticFD_batch_mode elasticFD_model_id_override ...
    elasticFD_source_id_override elasticFD_nt_override ...
    elasticFD_plot_times_override elasticFD_data_out_path ...
    elasticFD_record_out_path;

fprintf('Saved layered elastic data and record figure.\n');
