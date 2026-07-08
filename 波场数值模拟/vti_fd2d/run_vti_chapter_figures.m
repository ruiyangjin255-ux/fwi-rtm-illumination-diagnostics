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

runCase(3, 1, 1201, 201, 0.6, ...
    fullfile(figDir, 'isotropic_uniform_wavefield_0p6s_vx_vz.png'), ...
    fullfile(figDir, 'isotropic_uniform_record_vx_vz.png'), ...
    fullfile(dataDir, 'isotropic_uniform_data.mat'));

runCase(1, 1, 1201, 201, 0.6, ...
    fullfile(figDir, 'vti_uniform_wavefield_0p6s_vx_vz.png'), ...
    fullfile(figDir, 'vti_uniform_record_vx_vz.png'), ...
    fullfile(dataDir, 'vti_uniform_c13_1p5_data.mat'), 5);

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
    runCase(1, 1, 1001, 201, 0.6, ...
        fullfile(figDir, sprintf('vti_c13_%s_wavefield_0p6s_vx_vz.png', tag)), ...
        fullfile(figDir, sprintf('vti_c13_%s_record_vx_vz.png', tag)), ...
        fullfile(dataDir, sprintf('vti_c13_%s_data.mat', tag)), c13Index);
end

runCase(2, 1, 2001, 5, [0.6 1.0], ...
    fullfile(figDir, 'vti_layer_wavefield_vx_vz.png'), ...
    fullfile(figDir, 'vti_layer_record_2s_vx_vz.png'), ...
    fullfile(dataDir, 'vti_layer_data.mat'));

runCase(4, 1, 4001, 5, [1.0 1.5 2.0], ...
    fullfile(figDir, 'vti_salt_wavefield_vx_vz.png'), ...
    fullfile(figDir, 'vti_salt_record_4s_vx_vz.png'), ...
    fullfile(dataDir, 'vti_salt_data.mat'));

fprintf('VTI chapter simulations finished. Figures: %s\n', figDir);

function runCase(modelId, sourceId, ntValue, sourceDepth, plotTimes, snapshotPath, recordPath, dataPath, c13Index)
if nargin < 9
    c13Index = [];
end

vtiFD_batch_mode = true;
vtiFD_model_id_override = modelId;
vtiFD_source_id_override = sourceId;
vtiFD_nt_override = ntValue;
vtiFD_zs_orig_override = sourceDepth;
vtiFD_plot_times_override = plotTimes;
vtiFD_snapshot_out_path = snapshotPath;
vtiFD_record_out_path = recordPath;
vtiFD_data_out_path = dataPath;
vtiFD_no_total_title = true;

if ~isempty(c13Index)
    vtiFD_c13_index_override = c13Index;
end

run('D:\ryjin\vti_fd2d\vti_fd_chapter_data.m');
close all;
end

function tag = c13Tag(value)
if value < 0
    tag = sprintf('m%g', abs(value));
else
    tag = sprintf('p%g', value);
end
tag = strrep(tag, '.', 'p');
end
