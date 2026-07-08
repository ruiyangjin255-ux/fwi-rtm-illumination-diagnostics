close all; clc;

figDir = 'D:\ryjin\paper_figures_source\vti\c13_receiver_compare_432';
dataDir = 'D:\ryjin\vti_fd2d\saved_data\c13_receiver_compare_432';

if ~exist(figDir, 'dir')
    mkdir(figDir);
end
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

nx_orig = 401;
xs_orig = round((nx_orig + 1) / 2);

% Receivers are placed near the source on a dense line above it.
% The source is at (xs_orig, 201); this receiver line does not overlap it.
receiver_x_orig = (xs_orig - 100):(xs_orig + 100);
receiver_depth_orig = 181;

c13Cases = [
    2, -10.0
    4,   0.0
    5,   1.5
    7,  10.0
    8,  17.5
];

for ii = 1:size(c13Cases, 1)
    c13Index = c13Cases(ii, 1);
    c13Value = c13Cases(ii, 2);
    tag = c13Tag(c13Value);

    runCase(1, 1, 1001, 201, 0.6, receiver_x_orig, receiver_depth_orig, ...
        fullfile(figDir, sprintf('vti_c13_%s_wavefield_vx_vz.png', tag)), ...
        fullfile(figDir, sprintf('vti_c13_%s_near_source_record_1s_vx_vz.png', tag)), ...
        fullfile(dataDir, sprintf('vti_c13_%s_near_source_data.mat', tag)), c13Index);
end

makeReceiverPositionComparison(dataDir, figDir, c13Cases, receiver_x_orig, receiver_depth_orig, xs_orig);

fprintf('4.3.2 C13 receiver comparison finished. Figures: %s\n', figDir);

function runCase(modelId, sourceId, ntValue, sourceDepth, plotTimes, receiverX, receiverDepth, snapshotPath, recordPath, dataPath, c13Index)
vtiFD_batch_mode = true;
vtiFD_model_id_override = modelId;
vtiFD_source_id_override = sourceId;
vtiFD_nt_override = ntValue;
vtiFD_zs_orig_override = sourceDepth;
vtiFD_receiver_x_orig_override = receiverX;
vtiFD_receiver_depth_orig_override = receiverDepth;
vtiFD_plot_times_override = plotTimes;
vtiFD_snapshot_out_path = snapshotPath;
vtiFD_record_out_path = recordPath;
vtiFD_data_out_path = dataPath;
vtiFD_no_total_title = true;
vtiFD_wave_title_mode = 'component_only';
vtiFD_c13_index_override = c13Index;

run('D:\ryjin\vti_fd2d\vti_fd_chapter_data.m');
close all;
end

function makeReceiverPositionComparison(dataDir, figDir, c13Cases, receiverX, receiverDepth, xsOrig)
fig = figure('Visible','off','Color','w','Position',[100 100 1500 900]);
tl = tiledlayout(size(c13Cases, 1), 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for ii = 1:size(c13Cases, 1)
    c13Value = c13Cases(ii, 2);
    tag = c13Tag(c13Value);
    S = load(fullfile(dataDir, sprintf('vti_c13_%s_near_source_data.mat', tag)), ...
        'seis_vx', 'seis_vz', 't_axis', 'x_rec_km');

    [~, leftIdx] = min(abs(receiverX - (xsOrig - 40)));
    [~, nearLeftIdx] = min(abs(receiverX - (xsOrig - 10)));
    [~, nearRightIdx] = min(abs(receiverX - (xsOrig + 10)));
    [~, rightIdx] = min(abs(receiverX - (xsOrig + 40)));
    pickIdx = unique([leftIdx nearLeftIdx nearRightIdx rightIdx], 'stable');
    labels = arrayfun(@(x) sprintf('x=%.2f km', (x - 1) * 10 / 1000), receiverX(pickIdx), 'UniformOutput', false);

    nexttile;
    plotTraces(S.t_axis, S.seis_vx(:, pickIdx), labels);
    title(sprintf('Vx, c13=%.1f GPa', c13Value));

    nexttile;
    plotTraces(S.t_axis, S.seis_vz(:, pickIdx), labels);
    title(sprintf('Vz, c13=%.1f GPa', c13Value));
end

xlabel(tl, sprintf('Time (s), receiver depth %.2f km', (receiverDepth - 1) * 10 / 1000));
exportgraphics(fig, fullfile(figDir, 'vti_c13_near_source_trace_position_comparison.png'), 'Resolution', 300);
close(fig);
end

function plotTraces(t, traces, labels)
scale = max(abs(traces(:)));
if scale == 0
    scale = 1;
end
traces = traces / scale;
hold on;
offset = 0:numel(labels)-1;
for kk = 1:numel(labels)
    plot(t, traces(:,kk) + offset(kk), 'k', 'LineWidth', 0.9);
end
hold off;
grid on;
xlim([0 1]);
ylim([-0.8, numel(labels)-0.2]);
yticks(offset);
yticklabels(labels);
xlabel('Time (s)');
ylabel('Receiver');
set(gca, 'FontSize', 10, 'LineWidth', 0.8);
end

function tag = c13Tag(value)
if value < 0
    tag = sprintf('m%g', abs(value));
else
    tag = sprintf('p%g', value);
end
tag = strrep(tag, '.', 'p');
end
