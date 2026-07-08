close all; clc;

figDir = 'D:\ryjin\paper_figures_source\vti\thomsen_shape_scan_431';
dataDir = 'D:\ryjin\vti_fd2d\saved_data\thomsen_shape_scan_431';

if ~exist(figDir, 'dir'), mkdir(figDir); end
if ~exist(dataDir, 'dir'), mkdir(dataDir); end

% Keep the same Vx/Vz P-SV program. Only change Thomsen parameters.
cases = {
    'eps020_delta010', struct('vp0',2600,'vs0',1300,'rho',2400,'epsi',0.20,'delta',0.10,'gamma',0.00)
    'eps035_delta000', struct('vp0',2600,'vs0',1300,'rho',2400,'epsi',0.35,'delta',0.00,'gamma',0.00)
    'eps050_deltam015',struct('vp0',2600,'vs0',1300,'rho',2400,'epsi',0.50,'delta',-0.15,'gamma',0.00)
    'eps070_deltam025',struct('vp0',2600,'vs0',1300,'rho',2400,'epsi',0.70,'delta',-0.25,'gamma',0.00)
    'eps090_deltam035',struct('vp0',2600,'vs0',1300,'rho',2400,'epsi',0.90,'delta',-0.35,'gamma',0.00)
};

for ii = 1:size(cases, 1)
    tag = cases{ii, 1};
    th = cases{ii, 2};
    fprintf('Running Thomsen case: %s\n', tag);
    runOne(tag, th, figDir, dataDir);
end

makePanel(figDir, cases(:,1));
fprintf('Finished Thomsen shape scan. Figures: %s\n', figDir);

function runOne(tag, th, figDir, dataDir)
vtiFD_batch_mode = true;
vtiFD_model_id_override = 1;
vtiFD_source_id_override = 1;
vtiFD_thomsen_override = th;
vtiFD_nt_override = 1001;
vtiFD_f0_override = 10;
vtiFD_zs_orig_override = 201;
vtiFD_receiver_depth_orig_override = 181;
vtiFD_plot_times_override = 0.6;
vtiFD_fd_order_override = 6;
vtiFD_boundary_type_override = 'cpml';
vtiFD_no_total_title = true;
vtiFD_snapshot_out_path = fullfile(figDir, sprintf('%s_wavefield_0p6s_vx_vz.png', tag));
vtiFD_record_out_path = fullfile(figDir, sprintf('%s_record_1s_vx_vz.png', tag));
vtiFD_data_out_path = fullfile(dataDir, sprintf('%s_data.mat', tag));

run('D:\ryjin\vti_fd2d\vti_fd_chapter_data.m');
close all;
end

function makePanel(figDir, tags)
fig = figure('Visible','off','Color','w','Position',[80 80 1400 1700]);
tl = tiledlayout(numel(tags), 2, 'TileSpacing','compact', 'Padding','compact');

for ii = 1:numel(tags)
    S = load(fullfile('D:\ryjin\vti_fd2d\saved_data\thomsen_shape_scan_431', sprintf('%s_data.mat', tags{ii})), ...
        'vx_snaps', 'vz_snaps', 'x_km', 'z_km');

    nexttile;
    plotSnap(S.x_km, S.z_km, S.vx_snaps{1}, sprintf('%s  Vx', tags{ii}));
    nexttile;
    plotSnap(S.x_km, S.z_km, S.vz_snaps{1}, sprintf('%s  Vz', tags{ii}));
end

exportgraphics(fig, fullfile(figDir, 'thomsen_parameter_wavefield_scan_panel.png'), 'Resolution', 220);
close(fig);
end

function plotSnap(x, z, a, ttl)
imagesc(x, z, a);
set(gca, 'YDir', 'reverse');
axis image;
colormap(gca, seismicMap(256));
clipv = robustAbsClip(a, 0.995);
caxis([-clipv clipv]);
title(strrep(ttl, '_', '\_'), 'FontSize', 9);
xlabel('Distance (km)');
ylabel('Depth (km)');
set(gca, 'FontSize', 8, 'LineWidth', 0.8, 'Box','on');
end

function cmap = seismicMap(n)
half = floor(n/2);
blue = [linspace(0,1,half)' linspace(0,1,half)' ones(half,1)];
red = [ones(n-half,1) linspace(1,0,n-half)' linspace(1,0,n-half)'];
cmap = [blue; red];
end

function clipv = robustAbsClip(A, ratio)
v = sort(abs(A(:)));
idx = max(1, min(numel(v), round(ratio * numel(v))));
clipv = v(idx);
if clipv <= 0, clipv = max(v); end
if clipv <= 0, clipv = 1; end
end
