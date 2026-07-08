close all; clc;

dataDir = 'D:\ryjin\elastic_fd2d\saved_data';
figDir = 'D:\ryjin\paper_figures_source\elastic';
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

sourceTags = {'explosion', 'x_force', 'z_force', 'shear'};
sourceNames = {'Explosion source', 'Horizontal force source', ...
    'Vertical force source', 'Shear source'};

for ii = 1:numel(sourceTags)
    dataPath = fullfile(dataDir, sprintf('uniform_%s_record_data.mat', sourceTags{ii}));
    if ~exist(dataPath, 'file')
        error('Missing data file: %s', dataPath);
    end
    s = load(dataPath, 'seis_vx', 'seis_vz', 'x_km', 't_axis');
    outPath = fullfile(figDir, sprintf('uniform_%s_record_vx_vz.png', sourceTags{ii}));
    plotRecordPairSharedScale(s.seis_vx, s.seis_vz, s.x_km, s.t_axis, sourceNames{ii}, outPath);
end

function plotRecordPairSharedScale(seisVx, seisVz, xKm, tAxis, sourceName, outPath)
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

sgtitle(fig, sourceName, 'FontSize', 14, 'FontWeight', 'bold');
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
