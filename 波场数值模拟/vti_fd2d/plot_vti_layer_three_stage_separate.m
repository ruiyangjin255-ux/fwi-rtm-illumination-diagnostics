close all; clc;

dataPath = 'D:\ryjin\vti_fd2d\saved_data\vti_layer_three_stage_data.mat';
outDir = 'D:\ryjin\paper_figures_source\vti\layer_three_stage';

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

S = load(dataPath);

m = 256;
seismic_cmap = [linspace(0,1,m/2)', linspace(0,1,m/2)', ones(m/2,1); ...
                ones(m/2,1), linspace(1,0,m/2)', linspace(1,0,m/2)'];

stageNames = { ...
    'incident_qp_before_interface', ...
    'interface_reflection_transmission_conversion', ...
    'developed_reflection_transmission_conversion', ...
    'later_developed_reflection_transmission_conversion' ...
};

for k = 1:numel(S.plot_times)
    vx = S.vx_snaps{k};
    vz = S.vz_snaps{k};
    clipVal = localRobustClip([vx(:); vz(:)], 99.2);

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1280, 520]);
    tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    imagesc(S.x_km, S.z_km, vx);
    set(gca, 'YDir', 'reverse', 'FontSize', 14, 'LineWidth', 1.0);
    axis image;
    colormap(gca, seismic_cmap);
    caxis([-clipVal, clipVal]);
    title('Vx', 'FontSize', 18, 'FontWeight', 'bold');
    xlabel('Distance (km)', 'FontSize', 16);
    ylabel('Depth (km)', 'FontSize', 16);
    colorbar;

    nexttile;
    imagesc(S.x_km, S.z_km, vz);
    set(gca, 'YDir', 'reverse', 'FontSize', 14, 'LineWidth', 1.0);
    axis image;
    colormap(gca, seismic_cmap);
    caxis([-clipVal, clipVal]);
    title('Vz', 'FontSize', 18, 'FontWeight', 'bold');
    xlabel('Distance (km)', 'FontSize', 16);
    ylabel('Depth (km)', 'FontSize', 16);
    colorbar;

    outName = sprintf('vti_layer_stage%d_%s_t%03dms.png', ...
        k, stageNames{k}, round(S.plot_times(k) * 1000));
    exportgraphics(fig, fullfile(outDir, outName), 'Resolution', 300);
    close(fig);
end

fprintf('Separate layered VTI wavefield figures saved to: %s\n', outDir);

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
