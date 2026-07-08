close all; clc;

srcScript = 'D:\ryjin\vti_fd2d\vti_fd_chapter_data.m';
figDir = 'D:\ryjin\paper_figures_source\vti\uniform_thomsen_431';
dataDir = 'D:\ryjin\vti_fd2d\saved_data\uniform_thomsen_431';

if ~exist(figDir, 'dir')
    mkdir(figDir);
end
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

cases = struct( ...
    'name', {'fd4_pml', 'fd6_pml', 'fd4_cpml', 'fd6_cpml'}, ...
    'fdOrder', {4, 6, 4, 6}, ...
    'boundary', {'pml', 'pml', 'cpml', 'cpml'});

for ic = 1:numel(cases)
    clearvars -except srcScript figDir dataDir cases ic

    vtiFD_batch_mode = true;
    vtiFD_model_id_override = 1;
    vtiFD_source_id_override = 1;
    vtiFD_nt_override = 1001;
    vtiFD_zs_orig_override = 201;
    vtiFD_receiver_depth_orig_override = 205;
    vtiFD_plot_times_override = 0.6;
    vtiFD_fd_order_override = cases(ic).fdOrder;
    vtiFD_boundary_type_override = cases(ic).boundary;
    vtiFD_wave_title_mode = 'component_only';
    vtiFD_no_total_title = true;

    vtiFD_thomsen_override = struct( ...
        'vp0', 2600, ...
        'vs0', 1300, ...
        'rho', 2400, ...
        'epsi', 0.20, ...
        'delta', 0.10, ...
        'gamma', 0.0);

    vtiFD_snapshot_out_path = fullfile(figDir, ...
        ['uniform_vti_thomsen_' cases(ic).name '_wavefield_0p6s_vx_vz.png']);
    vtiFD_record_out_path = fullfile(figDir, ...
        ['uniform_vti_thomsen_' cases(ic).name '_record_1s_vx_vz.png']);
    vtiFD_data_out_path = fullfile(dataDir, ...
        ['uniform_vti_thomsen_' cases(ic).name '_data.mat']);

    run(srcScript);
end

fprintf('Uniform Thomsen VTI 4.3.1 simulations finished. Figures: %s\n', figDir);
