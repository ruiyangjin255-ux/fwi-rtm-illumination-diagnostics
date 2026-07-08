close all; clc;

srcScript = 'D:\ryjin\vti_fd2d\vti_fd_chapter_data.m';
figDir = 'D:\ryjin\paper_figures_source\vti';
dataDir = 'D:\ryjin\vti_fd2d\saved_data';

if ~exist(figDir, 'dir')
    mkdir(figDir);
end
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

vtiFD_batch_mode = true;
vtiFD_model_id_override = 1;
vtiFD_source_id_override = 1;
vtiFD_nt_override = 1201;
vtiFD_zs_orig_override = 201;
vtiFD_receiver_depth_orig_override = 205;
vtiFD_plot_times_override = 0.6;
vtiFD_snapshot_out_path = fullfile(figDir, 'vti_uniform_thomsen_wavefield_0p6s_vx_vz.png');
vtiFD_record_out_path = fullfile(figDir, 'vti_uniform_thomsen_record_vx_vz.png');
vtiFD_data_out_path = fullfile(dataDir, 'vti_uniform_thomsen_data.mat');
vtiFD_no_total_title = true;
vtiFD_thomsen_override = struct( ...
    'vp0', 2590.8852863768743, ...
    'vs0', 1435.1660156711187, ...
    'rho', 2680, ...
    'epsi', 0.19816564758198998, ...
    'delta', -0.23674495275598628, ...
    'gamma', 0.029891304347826088);

run(srcScript);

fprintf('Uniform Thomsen VTI simulation finished. Figures: %s\n', figDir);
