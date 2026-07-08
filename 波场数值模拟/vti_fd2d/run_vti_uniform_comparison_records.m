close all; clc;

srcScript = 'D:\ryjin\vti_fd2d\vti_fd_chapter_data.m';
figDir = 'D:\ryjin\paper_figures_source\vti\uniform_comparison_records';
dataDir = 'D:\ryjin\vti_fd2d\saved_data\uniform_comparison_records';

if ~exist(figDir, 'dir')
    mkdir(figDir);
end
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

thomsen = struct( ...
    'vp0', 2600, ...
    'vs0', 1300, ...
    'rho', 2400, ...
    'epsi', 0.20, ...
    'delta', 0.10, ...
    'gamma', 0.0);

%% 1. Differential-order comparison: fixed CPML, higher source frequency.
run_case(srcScript, dataDir, figDir, 'fd4_cpml_20hz_1p2s', 4, 'cpml', 20, 1201, thomsen);
run_case(srcScript, dataDir, figDir, 'fd6_cpml_20hz_1p2s', 6, 'cpml', 20, 1201, thomsen);

fd4 = load(fullfile(dataDir, 'fd4_cpml_20hz_1p2s_data.mat'));
fd6 = load(fullfile(dataDir, 'fd6_cpml_20hz_1p2s_data.mat'));
plot_three_record_panel(fd4, fd6, ...
    '4th-order + CPML', '6th-order + CPML', '4th - 6th residual', ...
    fullfile(figDir, 'uniform_vti_fd_order_record_residual_20hz.png'));

%% 2. Boundary comparison: fixed 6th-order FD, long record.
run_case(srcScript, dataDir, figDir, 'fd6_pml_20hz_4s', 6, 'pml', 20, 4001, thomsen);
run_case(srcScript, dataDir, figDir, 'fd6_cpml_20hz_4s', 6, 'cpml', 20, 4001, thomsen);

pml = load(fullfile(dataDir, 'fd6_pml_20hz_4s_data.mat'));
cpml = load(fullfile(dataDir, 'fd6_cpml_20hz_4s_data.mat'));
plot_three_record_panel(pml, cpml, ...
    '6th-order + PML', '6th-order + CPML', 'PML - CPML residual', ...
    fullfile(figDir, 'uniform_vti_boundary_record_residual_20hz_4s.png'));
plot_boundary_pair_enhanced(pml, cpml, ...
    fullfile(figDir, 'uniform_vti_boundary_record_vx_enhanced_20hz_4s.png'), 'vx');
plot_boundary_pair_enhanced(pml, cpml, ...
    fullfile(figDir, 'uniform_vti_boundary_record_vz_enhanced_20hz_4s.png'), 'vz');

fprintf('Uniform VTI record comparison figures finished. Figures: %s\n', figDir);

function run_case(srcScript, dataDir, figDir, name, fdOrder, boundary, f0, nt, thomsen)
    fprintf('\nRunning %s...\n', name);
    dataPath = fullfile(dataDir, [name '_data.mat']);
    if exist(dataPath, 'file')
        fprintf('Using existing data: %s\n', dataPath);
        return;
    end
    vtiFD_batch_mode = true;
    vtiFD_model_id_override = 1;
    vtiFD_source_id_override = 1;
    vtiFD_nt_override = nt;
    vtiFD_f0_override = f0;
    vtiFD_zs_orig_override = 201;
    vtiFD_receiver_depth_orig_override = 205;
    vtiFD_plot_times_override = 0.6;
    vtiFD_fd_order_override = fdOrder;
    vtiFD_boundary_type_override = boundary;
    vtiFD_wave_title_mode = 'component_only';
    vtiFD_no_total_title = true;
    vtiFD_thomsen_override = thomsen;
    vtiFD_snapshot_out_path = fullfile(figDir, [name '_wavefield_0p6s_vx_vz.png']);
    vtiFD_record_out_path = fullfile(figDir, [name '_record_vx_vz.png']);
    vtiFD_data_out_path = dataPath;
    run(srcScript);
end

function plot_three_record_panel(a, b, titleA, titleB, titleR, outPath)
    x = a.x_rec_km;
    t = a.t_axis;

    avx = prepare_record(a.seis_vx, t);
    bvx = prepare_record(b.seis_vx, t);
    avz = prepare_record(a.seis_vz, t);
    bvz = prepare_record(b.seis_vz, t);

    rvx = avx - bvx;
    rvz = avz - bvz;

    clip_vx = robust_abs_clip_local([avx(:); bvx(:)], 0.995);
    clip_vz = robust_abs_clip_local([avz(:); bvz(:)], 0.995);
    clip_rvx = robust_abs_clip_local(rvx(:), 0.995);
    clip_rvz = robust_abs_clip_local(rvz(:), 0.995);

    fig = figure('Visible','off','Color','w', 'Position', [80, 80, 1800, 980]);
    tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    plot_record_tile(x, t, avx, clip_vx, [titleA ' - Vx'], true, false);
    plot_record_tile(x, t, bvx, clip_vx, [titleB ' - Vx'], false, false);
    plot_record_tile(x, t, rvx, clip_rvx, [titleR ' - Vx'], false, false);

    plot_record_tile(x, t, avz, clip_vz, [titleA ' - Vz'], true, true);
    plot_record_tile(x, t, bvz, clip_vz, [titleB ' - Vz'], false, true);
    plot_record_tile(x, t, rvz, clip_rvz, [titleR ' - Vz'], false, true);
    exportgraphics(fig, outPath, 'Resolution', 300);
    close(fig);
end

function D = prepare_record(D, t)
    tgain = (t(:) + 0.02).^1.0;
    D = D .* tgain;
    amp = max(abs(D(:)));
    if amp == 0
        amp = 1;
    end
    D = D / amp;
end

function plot_boundary_pair_enhanced(pml, cpml, outPath, component)
    x = pml.x_rec_km;
    t = pml.t_axis;

    switch lower(component)
        case 'vx'
            A = prepare_record_enhanced(pml.seis_vx, t);
            B = prepare_record_enhanced(cpml.seis_vx, t);
        case 'vz'
            A = prepare_record_enhanced(pml.seis_vz, t);
            B = prepare_record_enhanced(cpml.seis_vz, t);
        otherwise
            error('Unsupported component: %s', component);
    end

    clipv = robust_abs_clip_local([A(:); B(:)], 0.985);

    fig = figure('Visible','off','Color','w', 'Position', [80, 80, 1450, 620]);
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    plot_record_tile_enhanced(x, t, A, clipv, 'PML boundary');
    plot_record_tile_enhanced(x, t, B, clipv, 'CPML boundary');

    exportgraphics(fig, outPath, 'Resolution', 600);
    close(fig);
end

function D = prepare_record_enhanced(D, t)
    D = D - mean(D, 1);
    tgain = (t(:) + 0.03).^1.25;
    D = D .* tgain;

    rms_trace = sqrt(mean(D.^2, 1));
    rms_trace(rms_trace == 0) = 1;
    D = D ./ rms_trace;

    amp = robust_abs_clip_local(D(:), 0.995);
    if amp == 0
        amp = 1;
    end
    D = D / amp;
end

function plot_record_tile_enhanced(x, t, D, clipv, ttl)
    nexttile;
    imagesc(x, t, D);
    colormap(gca, gray);
    set(gca, 'YDir', 'reverse', 'XAxisLocation', 'bottom', ...
        'FontSize', 13, 'LineWidth', 1.0, 'TickDir', 'in');
    caxis([-clipv clipv]);
    title(ttl, 'FontSize', 15, 'FontWeight', 'bold');
    xlabel('Distance (km)', 'FontSize', 14);
    ylabel('Time (s)', 'FontSize', 14);
    xlim([min(x) max(x)]);
    ylim([0 max(t)]);
    xticks(0:0.8:max(x));
    yticks(0:0.5:max(t));
    axis tight;
end

function plot_record_tile(x, t, D, clipv, ttl, showY, showX)
    nexttile;
    imagesc(x, t, D);
    colormap(gca, gray);
    set(gca, 'YDir', 'reverse', 'XAxisLocation', 'bottom', 'FontSize', 11);
    caxis([-clipv clipv]);
    title(ttl, 'FontSize', 12, 'FontWeight', 'normal');
    if showY
        ylabel('Time (s)');
    else
        yticklabels([]);
    end
    if showX
        xlabel('Distance (km)');
    else
        xlabel('');
    end
    if max(t) <= 1.5
        yticks(0:0.2:max(t));
    else
        yticks(0:0.5:max(t));
    end
end

function clipv = robust_abs_clip_local(A, ratio)
    v = sort(abs(A(:)));
    if isempty(v)
        clipv = 1;
        return;
    end
    idx = max(1, min(numel(v), round(ratio * numel(v))));
    clipv = v(idx);
    if clipv <= 0
        clipv = max(v);
    end
    if clipv <= 0
        clipv = 1;
    end
end
