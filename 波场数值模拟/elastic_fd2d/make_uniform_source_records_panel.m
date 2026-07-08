close all; clc;

dataDir = 'D:\ryjin\elastic_fd2d\saved_data';
outDir = 'D:\ryjin\paper_figures_source\elastic';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

sourceTags = {'explosion', 'x_force', 'z_force', 'shear'};
sourceNames = {'(a) Explosion source', '(b) Horizontal force source', ...
    '(c) Vertical force source', '(d) Shear source'};
recordsVx = cell(numel(sourceTags), 1);
recordsVz = cell(numel(sourceTags), 1);

for ii = 1:numel(sourceTags)
    dataPath = fullfile(dataDir, sprintf('uniform_%s_record_data.mat', sourceTags{ii}));
    if ~exist(dataPath, 'file')
        error('Missing saved data file: %s', dataPath);
    end
    s = load(dataPath, 'seis_vx', 'seis_vz', 'x_km', 't_axis');
    recordsVx{ii} = s.seis_vx;
    recordsVz{ii} = s.seis_vz;
    xAxis = s.x_km;
    tAxis = s.t_axis;
end

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1800 980]);

left = 0.055;
right = 0.025;
bottom = 0.08;
top = 0.035;
groupGapX = 0.075;
groupGapY = 0.04;
innerGap = 0.06;
groupW = (1 - left - right - groupGapX) / 2;
groupH = (1 - bottom - top - groupGapY) / 2;
axW = (groupW - innerGap) / 2;
labelH = 0.035;
labelGap = 0.055;
axH = groupH - labelH - labelGap;

for ii = 1:numel(sourceTags)
    row = floor((ii-1) / 2);
    col = mod(ii-1, 2);
    groupX = left + col * (groupW + groupGapX);
    groupY = 1 - top - (row + 1) * groupH - row * groupGapY;
    labelY = groupY;
    axY = groupY + labelH + labelGap;

    ax = axes(fig, 'Position', [groupX, axY, axW, axH]);
    imagesc(ax, xAxis, tAxis, recordsVx{ii});
    formatRecordAxis(ax, tAxis);
    xlabel(ax, 'Distance (km)', 'FontSize', 16);
    ylabel(ax, 'Time (s)', 'FontSize', 16);

    ax = axes(fig, 'Position', [groupX + axW + innerGap, axY, axW, axH]);
    imagesc(ax, xAxis, tAxis, recordsVz{ii});
    formatRecordAxis(ax, tAxis);
    xlabel(ax, 'Distance (km)', 'FontSize', 16);
    ylabel(ax, 'Time (s)', 'FontSize', 16);

    annotation(fig, 'textbox', [groupX, labelY, groupW, labelH], ...
        'String', sourceNames{ii}, 'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 18, 'FontWeight', 'bold');
end

exportgraphics(fig, fullfile(outDir, 'uniform_source_records_vx_vz_panel.png'), 'Resolution', 300);
close(fig);

function formatRecordAxis(ax, tAxis)
colormap(ax, gray(256));
set(ax, 'YDir', 'reverse', 'Box', 'on', 'Layer', 'top', ...
    'TickDir', 'in', 'FontSize', 14);
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
