close all; clc;

figDir = 'D:\ryjin\paper_figures_source\vti\layer_thomsen_near_receivers_by_time';
dataDir = 'D:\ryjin\vti_fd2d\saved_data';

if ~exist(figDir, 'dir')
    mkdir(figDir);
end
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

% Horizontal layered VTI model using vp0, vs0, density, epsilon and delta.
% Receivers cover the full model width and are placed slightly deeper
% than the surface source.
vtiFD_batch_mode = true;
vtiFD_model_id_override = 5;
vtiFD_source_id_override = 1;
vtiFD_nt_override = 2001;
vtiFD_zs_orig_override = 5;
vtiFD_receiver_depth_orig_override = 15;
vtiFD_receiver_x_orig_override = 1:401;
vtiFD_plot_times_override = [0.30 0.50 0.75 0.95];
vtiFD_boundary_type_override = 'cpml';
vtiFD_no_total_title = true;
vtiFD_wave_title_mode = 'component_only';

vtiFD_snapshot_out_path = '';
vtiFD_record_out_path = fullfile(figDir, 'vti_layer_thomsen_near_receivers_record_2s_vx_vz.png');
vtiFD_data_out_path = fullfile(dataDir, 'vti_layer_thomsen_near_receivers_by_time_data.mat');

run('D:\ryjin\vti_fd2d\vti_fd_chapter_data.m');
close all;

run('D:\ryjin\vti_fd2d\plot_vti_layer_thomsen_by_time.m');

fprintf('Horizontal layered VTI single-time figures saved to: %s\n', figDir);
