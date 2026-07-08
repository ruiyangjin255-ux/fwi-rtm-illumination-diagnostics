close all; clc;

dataDir = 'D:\ryjin\vti_fd2d\saved_data\thomsen_shape_scan_431';
outDir  = 'D:\ryjin\paper_figures_source\vti\thomsen_shape_scan_431_replot';

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

tags = {
    'eps020_delta010'
    'eps035_delta000'
    'eps050_deltam015'
    'eps070_deltam025'
    'eps090_deltam035'
};

for i = 1:numel(tags)
    tag = tags{i};
    matPath = fullfile(dataDir, [tag '_data.mat']);
    S = load(matPath, 'vx_snaps', 'vz_snaps', 'x_km', 'z_km', ...
        'seis_vx', 'seis_vz', 'x_rec_km', 't_axis');

    % 1) Wavefield snapshot at saved time: titles only Vx/Vz (no "at t=...")
    figW = figure('Visible', 'off', 'Color', 'w', 'Position', [120 120 1280 520]);
    tlW = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    draw_wave(S.x_km, S.z_km, S.vx_snaps{1}, 'Vx');

    nexttile;
    draw_wave(S.x_km, S.z_km, S.vz_snaps{1}, 'Vz');

    exportgraphics(figW, fullfile(outDir, [tag '_wavefield_vx_vz.png']), 'Resolution', 300);
    close(figW);

    % 2) Shot gather: no annotations, only axis formatting
    [vxDisp, vzDisp] = normalize_record(S.seis_vx, S.seis_vz, S.t_axis);

    figR = figure('Visible', 'off', 'Color', 'w', 'Position', [120 120 1480 540]);
    tlR = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    imagesc(S.x_rec_km, S.t_axis, vxDisp);
    set(gca, 'YDir', 'reverse', 'XAxisLocation', 'bottom');
    colormap(gca, gray);
    caxis([-0.45, 0.45]);
    xlabel('Distance (km)');
    ylabel('Time (s)');
    title('Shot Gather - Vx');
    format_record_axes(S.x_rec_km, S.t_axis);

    nexttile;
    imagesc(S.x_rec_km, S.t_axis, vzDisp);
    set(gca, 'YDir', 'reverse', 'XAxisLocation', 'bottom');
    colormap(gca, gray);
    caxis([-0.45, 0.45]);
    xlabel('Distance (km)');
    ylabel('Time (s)');
    title('Shot Gather - Vz');
    format_record_axes(S.x_rec_km, S.t_axis);

    exportgraphics(figR, fullfile(outDir, [tag '_record_vx_vz.png']), 'Resolution', 300);
    close(figR);

    % Optional combined panel per T-parameter case
    figP = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 1600 980]);
    tlP = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    draw_wave(S.x_km, S.z_km, S.vx_snaps{1}, 'Vx');
    nexttile;
    draw_wave(S.x_km, S.z_km, S.vz_snaps{1}, 'Vz');

    nexttile;
    imagesc(S.x_rec_km, S.t_axis, vxDisp);
    set(gca, 'YDir', 'reverse', 'XAxisLocation', 'bottom');
    colormap(gca, gray); caxis([-0.45, 0.45]);
    xlabel('Distance (km)'); ylabel('Time (s)'); title('Shot Gather - Vx');
    format_record_axes(S.x_rec_km, S.t_axis);

    nexttile;
    imagesc(S.x_rec_km, S.t_axis, vzDisp);
    set(gca, 'YDir', 'reverse', 'XAxisLocation', 'bottom');
    colormap(gca, gray); caxis([-0.45, 0.45]);
    xlabel('Distance (km)'); ylabel('Time (s)'); title('Shot Gather - Vz');
    format_record_axes(S.x_rec_km, S.t_axis);

    title(tlP, strrep(tag, '_', '\_'), 'FontWeight', 'bold');
    exportgraphics(figP, fullfile(outDir, [tag '_wavefield_record_panel.png']), 'Resolution', 260);
    close(figP);
end

fprintf('Replot finished from saved data: %s\n', outDir);

function draw_wave(x, z, A, ttl)
imagesc(x, z, A);
set(gca, 'YDir', 'reverse');
axis image;
colormap(gca, seismic_map(256));
clipv = robust_abs_clip(A, 0.995);
caxis([-clipv clipv]);
xlabel('Distance (km)');
ylabel('Depth (km)');
title(ttl);
set(gca, 'FontSize', 12, 'LineWidth', 1.0, 'Box', 'on');
cb = colorbar;
set(cb, 'FontSize', 11);
end

function [vxDisp, vzDisp] = normalize_record(seis_vx, seis_vz, t)
tgain = (t(:) + 0.02).^1.0;
vxDisp = seis_vx .* tgain;
vzDisp = seis_vz .* tgain;

mx = max(abs(vxDisp(:))); if mx == 0, mx = 1; end
mz = max(abs(vzDisp(:))); if mz == 0, mz = 1; end
vxDisp = vxDisp / mx;
vzDisp = vzDisp / mz;
end

function format_record_axes(xrec, t)
xmin = min(xrec); xmax = max(xrec);
xlim([xmin xmax]);
xt = ceil(xmin):1:floor(xmax);
if isempty(xt)
    xt = linspace(xmin, xmax, 5);
end
xticks(xt);
xticklabels(compose('%.0f', xt));

tmax = max(t);
yt = 0:0.5:tmax;
yticks(yt);
yticklabels(compose('%.1f', yt));
set(gca, 'FontSize', 12, 'LineWidth', 1.0, 'Box', 'on');
end

function clipv = robust_abs_clip(A, ratio)
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

function cmap = seismic_map(n)
half = floor(n/2);
blue = [linspace(0,1,half)', linspace(0,1,half)', ones(half,1)];
red  = [ones(n-half,1), linspace(1,0,n-half)', linspace(1,0,n-half)'];
cmap = [blue; red];
end
