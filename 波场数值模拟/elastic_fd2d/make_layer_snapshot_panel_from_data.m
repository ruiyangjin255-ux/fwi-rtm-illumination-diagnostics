close all; clc;

dataPath = 'D:\ryjin\elastic_fd2d\saved_data\layer_explosion_record_snapshots_data.mat';
outDir = 'D:\ryjin\paper_figures_source\elastic';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
if ~exist(dataPath, 'file')
    error('Missing saved data file: %s', dataPath);
end

s = load(dataPath, 'vx_snaps', 'vz_snaps', 'x_km', 'z_km', 'plot_times');
plotTimes = s.plot_times;

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1200 2200]);
cmap = seismicMap(256);

left = 0.08;
right = 0.035;
bottom = 0.055;
top = 0.035;
colGap = 0.075;
rowGap = 0.035;
axW = (1 - left - right - colGap) / 2;
axH = (1 - bottom - top - 3 * rowGap) / 4;

for ii = 1:numel(plotTimes)
    vx = s.vx_snaps{ii};
    vz = s.vz_snaps{ii};
    lim = max([max(abs(vx(:))), max(abs(vz(:)))]);
    if lim <= 0
        lim = 1;
    end
    lim = 0.5 * lim;

    axY = 1 - top - ii * axH - (ii - 1) * rowGap;
    ax = axes(fig, 'Position', [left, axY, axW, axH]);
    imagesc(ax, s.x_km, s.z_km, vx);
    formatSnapshotAxis(ax, cmap, lim);
    ylabel(ax, 'Depth (km)', 'FontSize', 14);
    if ii == numel(plotTimes)
        xlabel(ax, 'Distance (km)', 'FontSize', 14);
    else
        ax.XTickLabel = [];
    end
    text(ax, 0.03, 0.08, sprintf('Vx at t=%.1fs', plotTimes(ii)), ...
        'Units', 'normalized', 'FontSize', 13, 'FontWeight', 'bold', ...
        'BackgroundColor', [1 1 1], 'Margin', 2);

    ax = axes(fig, 'Position', [left + axW + colGap, axY, axW, axH]);
    imagesc(ax, s.x_km, s.z_km, vz);
    formatSnapshotAxis(ax, cmap, lim);
    ylabel(ax, 'Depth (km)', 'FontSize', 14);
    if ii == numel(plotTimes)
        xlabel(ax, 'Distance (km)', 'FontSize', 14);
    else
        ax.XTickLabel = [];
    end
    text(ax, 0.03, 0.08, sprintf('Vz at t=%.1fs', plotTimes(ii)), ...
        'Units', 'normalized', 'FontSize', 13, 'FontWeight', 'bold', ...
        'BackgroundColor', [1 1 1], 'Margin', 2);
end

exportgraphics(fig, fullfile(outDir, 'layer_wavefield_snapshots_0p6_0p8_1p0_1p2_vx_vz.png'), ...
    'Resolution', 300);
close(fig);

function formatSnapshotAxis(ax, cmap, lim)
set(ax, 'YDir', 'reverse', 'Box', 'on', 'Layer', 'top', ...
    'TickDir', 'in', 'FontSize', 12);
axis(ax, 'image');
colormap(ax, cmap);
clim(ax, [-lim lim]);
xlim(ax, [0 4]);
ylim(ax, [0 4]);
xticks(ax, 0:1:4);
yticks(ax, 0:1:4);
end

function cmap = seismicMap(n)
if nargin < 1
    n = 256;
end
half = floor(n/2);
blue = [linspace(0,1,half)' linspace(0,1,half)' ones(half,1)];
red = [ones(n-half,1) linspace(1,0,n-half)' linspace(1,0,n-half)'];
cmap = [blue; red];
end
