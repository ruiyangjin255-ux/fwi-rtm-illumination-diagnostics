function replot_elastic_figures_from_bins()
close all; clc;

outDir = 'D:\ryjin\paper_figures_source\elastic';
resDir = 'D:\ryjin\elastic_fd2d\results';
dx = 10; dz = 10; dt = 0.001;
nx = 401; nz = 401; nt = 2000;
plotTimes = [0.6 0.8 1.0];
xKm = (0:nx-1) * dx / 1000;
zKm = (0:nz-1) * dz / 1000;
tAxis = (0:nt-1) * dt;

models = {
    struct('key','uniform','label','Homogeneous','src',[201 2]), ...
    struct('key','layer','label','Layered','src',[201 2])
};

for i = 1:numel(models)
    m = models{i};
    vp = readFloatBin(fullfile(resDir, [m.key '_vp.bin']), [nz nx]);
    plotElasticModel(vp, xKm, zKm, m, fullfile(outDir, [m.key '_elastic_model.png']));

    seisVx = readFloatBin(fullfile(resDir, [m.key '_record_vx.bin']), [nt nx]);
    seisVz = readFloatBin(fullfile(resDir, [m.key '_record_vz.bin']), [nt nx]);
    plotShotGatherPair(seisVx, seisVz, xKm, tAxis, fullfile(outDir, [m.key '_record_vx_vz.png']));

    for k = 1:numel(plotTimes)
        suffix = timeSuffix(plotTimes(k));
        vx = readFloatBin(fullfile(resDir, sprintf('%s_vx_%ss.bin', m.key, suffix)), [nz nx]);
        vz = readFloatBin(fullfile(resDir, sprintf('%s_vz_%ss.bin', m.key, suffix)), [nz nx]);
        plotSnapshotPair(vx, vz, xKm, zKm, plotTimes(k), ...
            fullfile(outDir, sprintf('%s_wavefield_%ss_vx_vz.png', m.key, suffix)));
    end
end
fprintf('Replotted elastic figures without super titles under %s\n', outDir);
end

function plotSnapshotPair(vx, vz, xKm, zKm, timeValue, outPath)
cmap = seismicMap(256);
lim = percentile(abs([vx(:); vz(:)]), 99.5);
if lim <= 0, lim = 1; end
lim = 0.55 * lim;
fig = figure('Visible','off','Color','w','Position',[100 100 1100 420]);
tiledlayout(fig, 1, 2, 'TileSpacing','compact', 'Padding','compact');
ax = nexttile;
imagesc(ax, xKm, zKm, vx); set(ax,'YDir','reverse'); axis(ax,'image');
colormap(ax, cmap); clim(ax, [-lim lim]); colorbar(ax);
title(ax, sprintf('Vx at t=%.1fs', timeValue), 'FontSize',13);
xlabel(ax, 'Distance (km)'); ylabel(ax, 'Depth (km)');
ax = nexttile;
imagesc(ax, xKm, zKm, vz); set(ax,'YDir','reverse'); axis(ax,'image');
colormap(ax, cmap); clim(ax, [-lim lim]); colorbar(ax);
title(ax, sprintf('Vz at t=%.1fs', timeValue), 'FontSize',13);
xlabel(ax, 'Distance (km)'); ylabel(ax, 'Depth (km)');
exportgraphics(fig, outPath, 'Resolution', 300);
close(fig);
end

function plotShotGatherPair(seisVx, seisVz, xKm, tAxis, outPath)
seisVx = seisVx ./ max(max(abs(seisVx)), eps);
seisVz = seisVz ./ max(max(abs(seisVz)), eps);
fig = figure('Visible','off','Color','w','Position',[100 100 1000 450]);
ax = subplot(1,2,1);
imagesc(ax, xKm, tAxis, seisVx); set(ax,'YDir','reverse'); colormap(ax, gray(256));
clim(ax, [-0.08 0.08]); title(ax, 'Shot Gather - Vx', 'FontSize',13);
xlabel(ax, 'Distance (km)'); ylabel(ax, 'Time (s)');
ax = subplot(1,2,2);
imagesc(ax, xKm, tAxis, seisVz); set(ax,'YDir','reverse'); colormap(ax, gray(256));
clim(ax, [-0.08 0.08]); title(ax, 'Shot Gather - Vz', 'FontSize',13);
xlabel(ax, 'Distance (km)'); ylabel(ax, 'Time (s)');
exportgraphics(fig, outPath, 'Resolution', 300);
close(fig);
end

function plotElasticModel(vp, xKm, zKm, m, outPath)
fig = figure('Visible','off','Color','w','Position',[100 100 620 560]);
ax = axes(fig);
imagesc(ax, xKm, zKm, vp/1000);
set(ax,'YDir','reverse'); axis(ax,'image'); colormap(ax, jet(256));
cb = colorbar(ax); cb.Label.String = 'V_p (km/s)';
title(ax, sprintf('Elastic Model - %s', m.label), 'FontSize',14, 'FontWeight','bold');
xlabel(ax, 'Distance (km)'); ylabel(ax, 'Depth (km)');
hold(ax, 'on');
plot(ax, xKm(m.src(1)), zKm(m.src(2)), ...
    'p', 'MarkerSize',12, 'MarkerFaceColor','r', 'MarkerEdgeColor','k');
exportgraphics(fig, outPath, 'Resolution', 300);
close(fig);
end

function a = readFloatBin(path, shape)
fid = fopen(path, 'rb');
if fid < 0, error('Cannot open %s', path); end
a = fread(fid, prod(shape), 'single=>single');
fclose(fid);
if numel(a) ~= prod(shape)
    error('Unexpected file size: %s', path);
end
a = reshape(a, shape);
end

function cmap = seismicMap(n)
if nargin < 1, n = 256; end
half = floor(n/2);
blue = [linspace(0,1,half)' linspace(0,1,half)' ones(half,1)];
red = [ones(n-half,1) linspace(1,0,n-half)' linspace(1,0,n-half)'];
cmap = [blue; red];
end

function p = percentile(x, q)
x = sort(double(x(:)));
if isempty(x), p = 0; return; end
idx = max(1, min(numel(x), round(q/100 * numel(x))));
p = x(idx);
end

function suffix = timeSuffix(t)
suffix = strrep(sprintf('%.1f', t), '.', 'p');
end
