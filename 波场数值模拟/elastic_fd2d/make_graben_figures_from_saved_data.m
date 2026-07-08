close all; clc;

dataPath = 'D:\ryjin\elastic_fd2d\saved_data\graben_explosion_record_snapshots_4s_data.mat';
outDir = 'D:\ryjin\paper_figures_source\elastic';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
if ~exist(dataPath, 'file')
    error('Missing saved data file: %s', dataPath);
end

s = load(dataPath);
xKm = s.x_km;
zKm = s.z_km;
tAxis = s.t_axis;
cmap = seismicMap(256);

plotRecordPair(s.seis_vx, s.seis_vz, xKm, tAxis, ...
    fullfile(outDir, 'graben_record_vx_vz_4s.png'));

for ii = 1:numel(s.plot_times)
    timeLabel = strrep(sprintf('%.1f', s.plot_times(ii)), '.', 'p');
    plotSnapshotPair(s.vx_snaps{ii}, s.vz_snaps{ii}, xKm, zKm, cmap, s.plot_times(ii), ...
        fullfile(outDir, sprintf('graben_wavefield_%ss_vx_vz.png', timeLabel)));
end

function plotRecordPair(seisVx, seisVz, xKm, tAxis, outPath)
scale = max(abs([seisVx(:); seisVz(:)]));
if scale == 0
    scale = 1;
end
seisVx = seisVx ./ scale;
seisVz = seisVz ./ scale;

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1000 450]);

ax = subplot(1, 2, 1);
imagesc(ax, xKm, tAxis, seisVx);
formatRecordAxis(ax, tAxis);
xlabel(ax, 'Distance (km)');
ylabel(ax, 'Time (s)');
title(ax, 'Shot Gather - Vx', 'FontSize', 13, 'FontWeight', 'normal');

ax = subplot(1, 2, 2);
imagesc(ax, xKm, tAxis, seisVz);
formatRecordAxis(ax, tAxis);
xlabel(ax, 'Distance (km)');
ylabel(ax, 'Time (s)');
title(ax, 'Shot Gather - Vz', 'FontSize', 13, 'FontWeight', 'normal');

exportgraphics(fig, outPath, 'Resolution', 300);
close(fig);
end

function plotSnapshotPair(vx, vz, xKm, zKm, cmap, timeValue, outPath)
maxVal = max([max(abs(vx(:))), max(abs(vz(:)))]);
if maxVal == 0
    maxVal = 1;
end
clipVal = 0.5 * maxVal;

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 450]);

ax = subplot(1, 2, 1);
imagesc(ax, xKm, zKm, vx);
formatSnapshotAxis(ax, cmap, clipVal);
title(ax, sprintf('Vx at t=%.1fs', timeValue), 'FontSize', 13, 'FontWeight', 'normal');
xlabel(ax, 'Distance (km)');
ylabel(ax, 'Depth (km)');
colorbar(ax);

ax = subplot(1, 2, 2);
imagesc(ax, xKm, zKm, vz);
formatSnapshotAxis(ax, cmap, clipVal);
title(ax, sprintf('Vz at t=%.1fs', timeValue), 'FontSize', 13, 'FontWeight', 'normal');
xlabel(ax, 'Distance (km)');
ylabel(ax, 'Depth (km)');
colorbar(ax);

exportgraphics(fig, outPath, 'Resolution', 300);
close(fig);
end

function formatRecordAxis(ax, tAxis)
colormap(ax, gray(256));
set(ax, 'YDir', 'reverse', 'Box', 'on', 'Layer', 'top', ...
    'TickDir', 'in', 'FontSize', 12);
clim(ax, [-0.08 0.08]);
xlim(ax, [0 4]);
ylim(ax, [0 max(tAxis)]);
xticks(ax, 0:1:4);
yt = 0:0.1:max(tAxis);
yticks(ax, yt);
ytLabel = strings(size(yt));
for jj = 1:numel(yt)
    if abs(mod(yt(jj), 0.5)) < 1e-8 || abs(mod(yt(jj), 0.5) - 0.5) < 1e-8
        ytLabel(jj) = num2str(yt(jj), '%.1f');
    else
        ytLabel(jj) = "";
    end
end
yticklabels(ax, ytLabel);
end

function formatSnapshotAxis(ax, cmap, clipVal)
set(ax, 'YDir', 'reverse', 'FontSize', 12);
axis(ax, 'image');
colormap(ax, cmap);
clim(ax, [-clipVal clipVal]);
xlim(ax, [0 4]);
ylim(ax, [0 4]);
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
