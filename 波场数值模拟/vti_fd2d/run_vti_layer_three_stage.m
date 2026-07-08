close all; clc;

figDir = 'D:\ryjin\paper_figures_source\vti\layer_three_stage';
dataDir = 'D:\ryjin\vti_fd2d\saved_data';

if ~exist(figDir, 'dir')
    mkdir(figDir);
end
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

% Horizontal layered VTI closing experiment:
% 0.30 s: incident qP wave before strong interface interaction
% 0.50 s: wavefield begins reflection/transmission/conversion at interface
% 0.75 s: reflected, transmitted and converted waves are developed
% 0.95 s: later developed reflected, transmitted and converted waves
vtiFD_batch_mode = true;
vtiFD_model_id_override = 2;
vtiFD_source_id_override = 1;
vtiFD_nt_override = 2001;
vtiFD_zs_orig_override = 5;
vtiFD_receiver_depth_orig_override = 5;
vtiFD_plot_times_override = [0.30 0.50 0.75 0.95];
vtiFD_boundary_type_override = 'cpml';
vtiFD_no_total_title = true;
vtiFD_wave_title_mode = 'component_time';

vtiFD_snapshot_out_path = fullfile(figDir, 'vti_layer_three_stage_wavefield_vx_vz.png');
vtiFD_record_out_path = fullfile(figDir, 'vti_layer_three_stage_record_2s_vx_vz.png');
vtiFD_data_out_path = fullfile(dataDir, 'vti_layer_three_stage_data.mat');

run('D:\ryjin\vti_fd2d\vti_fd_chapter_data.m');
close all;

fprintf('Horizontal layered VTI three-stage figures saved to: %s\n', figDir);
