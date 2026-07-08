close all; clc;

figDir = 'D:\ryjin\paper_figures_source\vti\layer_thomsen_surface';
dataDir = 'D:\ryjin\vti_fd2d\saved_data';

if ~exist(figDir, 'dir')
    mkdir(figDir);
end
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

% Two-layer VTI model parameterized by vp0, vs0, density, epsilon and delta.
% Source and receivers are placed near the surface grid line.
vtiFD_batch_mode = true;
vtiFD_model_id_override = 5;
vtiFD_source_id_override = 1;
vtiFD_nt_override = 2001;
vtiFD_zs_orig_override = 5;
vtiFD_receiver_depth_orig_override = 5;
vtiFD_receiver_x_orig_override = 1:401;
vtiFD_plot_times_override = [0.30 0.50 0.75 0.95];
vtiFD_boundary_type_override = 'cpml';
vtiFD_no_total_title = true;
vtiFD_wave_title_mode = 'component_time';

vtiFD_snapshot_out_path = fullfile(figDir, 'vti_layer_thomsen_surface_wavefield_vx_vz.png');
vtiFD_record_out_path = fullfile(figDir, 'vti_layer_thomsen_surface_record_2s_vx_vz.png');
vtiFD_data_out_path = fullfile(dataDir, 'vti_layer_thomsen_surface_data.mat');

run('D:\ryjin\vti_fd2d\vti_fd_chapter_data.m');
close all;

fprintf('Two-layer VTI Thomsen surface-source figures saved to: %s\n', figDir);
