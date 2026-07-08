close all; clc;

if ~exist('vtiLayerDataPath', 'var') || isempty(vtiLayerDataPath)
    vtiLayerDataPath = 'D:\ryjin\vti_fd2d\saved_data\vti_layer_thomsen_near_receivers_by_time_data.mat';
end
if ~exist('vtiLayerOutDir', 'var') || isempty(vtiLayerOutDir)
    vtiLayerOutDir = 'D:\ryjin\paper_figures_source\vti\layer_thomsen_near_receivers_by_time';
end
if ~exist('vtiLayerNamePrefix', 'var') || isempty(vtiLayerNamePrefix)
    vtiLayerNamePrefix = 'vti_layer_thomsen_followup';
end

if ~exist(vtiLayerOutDir, 'dir')
    mkdir(vtiLayerOutDir);
end

S = load(vtiLayerDataPath);

m = 256;
seismic_cmap = [linspace(0,1,m/2)', linspace(0,1,m/2)', ones(m/2,1); ...
                ones(m/2,1), linspace(1,0,m/2)', linspace(1,0,m/2)'];

for k = 1:numel(S.plot_times)
    vx = S.vx_snaps{k};
    vz = S.vz_snaps{k};
    clipVal = localRobustClip([vx(:); vz(:)], 99.2);

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1280, 520]);
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    imagesc(S.x_km, S.z_km, vx);
    set(gca, 'YDir', 'reverse', 'FontSize', 14, 'LineWidth', 1.0);
    axis image;
    colormap(gca, seismic_cmap);
    caxis([-clipVal, clipVal]);
    title('Vx', 'FontSize', 18, 'FontWeight', 'normal');
    xlabel('Distance (km)', 'FontSize', 16);
    ylabel('Depth (km)', 'FontSize', 16);
    colorbar;

    nexttile;
    imagesc(S.x_km, S.z_km, vz);
    set(gca, 'YDir', 'reverse', 'FontSize', 14, 'LineWidth', 1.0);
    axis image;
    colormap(gca, seismic_cmap);
    caxis([-clipVal, clipVal]);
    title('Vz', 'FontSize', 18, 'FontWeight', 'normal');
    xlabel('Distance (km)', 'FontSize', 16);
    ylabel('Depth (km)', 'FontSize', 16);
    colorbar;

    outName = sprintf('%s_t%04dms_vx_vz.png', ...
        vtiLayerNamePrefix, round(S.plot_times(k) * 1000));
    exportgraphics(fig, fullfile(vtiLayerOutDir, outName), 'Resolution', 300);
    close(fig);
end

fprintf('Generic single-time wavefield figures exported to: %s\n', vtiLayerOutDir);

function clipVal = localRobustClip(data, pct)
    data = abs(data(:));
    data = data(isfinite(data));
    data = data(data > 0);
    if isempty(data)
        clipVal = 1;
    else
        clipVal = prctile(data, pct);
        if clipVal <= 0
            clipVal = max(data);
        end
    end
end
