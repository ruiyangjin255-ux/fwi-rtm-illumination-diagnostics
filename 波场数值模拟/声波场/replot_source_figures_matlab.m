function replot_source_figures_matlab()
% Replot source-program binary results for thesis figures.
% Numerical data come from fd2d_pml and elastic_fd2d program outputs.

close all;
set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultTextFontName', 'Arial');
set(groot, 'defaultAxesFontSize', 9);
set(groot, 'defaultTextFontSize', 9);

outRoot = 'D:\ryjin\paper_figures_source';
acousticOut = fullfile(outRoot, 'acoustic');
elasticOut = fullfile(outRoot, 'elastic');
ensureDir(acousticOut);
ensureDir(elasticOut);

plotAcoustic(acousticOut);
try
    plotElastic(elasticOut);
catch ME
    warning('Elastic figure replot skipped: %s', ME.message);
end
fprintf('Finished source-program figure replots under %s\n', outRoot);
end

function plotAcoustic(outDir)
root = 'D:\ryjin\acoustic_results_multitime';
models = {
    struct('key','uniform','title','Uniform acoustic model','nx',401,'nz',401,'nt',4001,'dx',0.01,'dz',0.01,'dt',0.001,'source',[200 200],'times',[0.2 0.4 0.6 0.8 1.0 1.2],'rows',2,'cols',3), ...
    struct('key','layer','title','Layered acoustic model','nx',401,'nz',401,'nt',4001,'dx',0.01,'dz',0.01,'dt',0.001,'source',[200 1],'times',[0.2 0.4 0.6 0.8 1.0 1.2],'rows',2,'cols',3), ...
    struct('key','graben','title','Graben acoustic model','nx',401,'nz',401,'nt',4001,'dx',0.01,'dz',0.01,'dt',0.001,'source',[200 1],'times',[0.4 0.8 1.2 1.4 1.6],'rows',3,'cols',2), ...
    struct('key','seg','title','SEG complex acoustic model','nx',676,'nz',230,'nt',4001,'dx',0.01,'dz',0.01,'dt',0.001,'source',[340 1],'times',[0.4 0.6 0.8 1.0],'rows',4,'cols',1)
};

for i = 1:numel(models)
    m = models{i};
    base = fullfile(root, m.key);
    vel = readBin(fullfile(base, [m.key '_velocity.bin']), [m.nz m.nx]);
    plotVelocityModel(vel/1000, m, fullfile(outDir, [m.key '_model_source_matlab.png']), 'Velocity (km/s)');

    snaps = cell(size(m.times));
    for k = 1:numel(m.times)
        suffix = timeSuffix(m.times(k));
        snaps{k} = readBin(fullfile(base, sprintf('%s_snapshot_%ss.bin', m.key, suffix)), [m.nz m.nx]);
    end
    plotWavefieldSet(snaps, m, fullfile(outDir, [m.key '_wavefield_multitime_matlab.png']));

    rec = readBin(fullfile(base, [m.key '_record.bin']), [m.nt m.nx]);
    plotRecord(rec, m, fullfile(outDir, [m.key '_record_matlab.png']));
end
end

function plotElastic(outDir)
root = 'D:\ryjin\elastic_fd2d\results';
models = {
    struct('key','uniform','title','Uniform elastic model','nx',401,'nz',401,'nt',3401,'dx',0.01,'dz',0.01,'dt',0.0005,'source',[200 200],'times',[0.4 0.8 1.2 1.4 1.6],'rows',3,'cols',2), ...
    struct('key','layer','title','Layered elastic model','nx',401,'nz',401,'nt',3401,'dx',0.01,'dz',0.01,'dt',0.0005,'source',[200 20],'times',[0.4 0.8 1.2 1.4 1.6],'rows',3,'cols',2), ...
    struct('key','fault','title','Fault elastic model','nx',401,'nz',401,'nt',3401,'dx',0.01,'dz',0.01,'dt',0.0005,'source',[200 20],'times',[0.4 0.8 1.2 1.4 1.6],'rows',3,'cols',2)
};

for i = 1:numel(models)
    m = models{i};
    vp = readBin(fullfile(root, [m.key '_vp.bin']), [m.nz m.nx]);
    vs = readBin(fullfile(root, [m.key '_vs.bin']), [m.nz m.nx]);
    plotVelocityModel(vp/1000, m, fullfile(outDir, [m.key '_vp_model_source_matlab.png']), 'Vp (km/s)');
    plotVelocityModel(vs/1000, m, fullfile(outDir, [m.key '_vs_model_source_matlab.png']), 'Vs (km/s)');

    vxSnaps = cell(size(m.times));
    vzSnaps = cell(size(m.times));
    for c = ["vx", "vz"]
        snaps = cell(size(m.times));
        for k = 1:numel(m.times)
            suffix = timeSuffix(m.times(k));
            snaps{k} = readBin(fullfile(root, sprintf('%s_%s_%ss.bin', m.key, c, suffix)), [m.nz m.nx]);
        end
        if c == "vx"
            vxSnaps = snaps;
        else
            vzSnaps = snaps;
        end
        m2 = m;
        m2.component = char(c);
        plotWavefieldSet(snaps, m2, fullfile(outDir, sprintf('%s_%s_wavefield_multitime_matlab.png', m.key, c)));

        rec = readBin(fullfile(root, sprintf('%s_record_%s.bin', m.key, c)), [m.nt m.nx]);
        plotRecord(rec, m, fullfile(outDir, sprintf('%s_%s_record_matlab.png', m.key, c)));
    end
    plotElasticPairSet(vxSnaps, vzSnaps, m, fullfile(outDir, sprintf('%s_vx_vz_main_times_matlab.png', m.key)));
end
end

function plotVelocityModel(model, m, outPath, cbLabel)
xmax = (m.nx - 1) * m.dx;
zmax = (m.nz - 1) * m.dz;
fig = figure('Visible','off','Position',[100 100 680 560]);
ax = axes(fig);
imagesc(ax, [0 xmax], [0 zmax], model);
axis(ax, 'image');
set(ax, 'YDir','reverse', 'FontSize',9, 'LineWidth',0.9, 'Box','on');
xlabel(ax, 'Distance (km)', 'FontSize',10);
ylabel(ax, 'Depth (km)', 'FontSize',10);
colormap(ax, jet(256));
cb = colorbar(ax);
cb.Label.String = cbLabel;
cb.Label.FontSize = 10;
cb.FontSize = 9;
caxis(ax, [1 4]);
hold(ax, 'on');
sx = m.source(1) * m.dx;
sz = m.source(2) * m.dz;
szMark = min(max(sz, 0.08), zmax - 0.08);
plot(ax, sx, szMark, 'p', 'MarkerSize',12, 'MarkerFaceColor','r', 'MarkerEdgeColor','k', 'LineWidth',0.7);
text(ax, min(sx + 0.10, xmax - 0.55), szMark + 0.06, 'Source', ...
    'FontSize',9, 'Color','k', 'FontWeight','bold', 'Clipping','on');
exportgraphics(fig, outPath, 'Resolution', 300);
close(fig);
end

function plotWavefieldSet(snaps, m, outPath)
rows = m.rows;
cols = m.cols;
if strcmp(m.key, 'seg')
    fig = figure('Visible','off','Position',[60 40 1120 1520]);
elseif numel(snaps) == 5 && rows == 3 && cols == 2
    fig = figure('Visible','off','Position',[80 40 1050 1280]);
elseif numel(snaps) == 6 && rows == 2 && cols == 3
    fig = figure('Visible','off','Position',[80 80 1180 760]);
else
    fig = figure('Visible','off','Position',[80 80 1080 780]);
end
letters = 'abcdef';
xmax = (m.nx - 1) * m.dx;
zmax = (m.nz - 1) * m.dz;
if strcmp(m.key, 'seg')
    panelPos = [
        0.10 0.800 0.80 0.150
        0.10 0.560 0.80 0.150
        0.10 0.320 0.80 0.150
        0.10 0.080 0.80 0.150
    ];
    captionGap = 0.055;
    useManualLayout = true;
elseif numel(snaps) == 5 && rows == 3 && cols == 2
    panelPos = [
        0.075 0.725 0.375 0.215
        0.555 0.725 0.375 0.215
        0.075 0.405 0.375 0.215
        0.555 0.405 0.375 0.215
        0.315 0.085 0.375 0.215
    ];
    captionGap = 0.050;
    useManualLayout = true;
else
    t = tiledlayout(fig, rows, cols, 'TileSpacing','compact', 'Padding','compact');
    useManualLayout = false;
end
for k = 1:numel(snaps)
    if useManualLayout
        ax = axes(fig, 'Position', panelPos(k, :));
    else
        ax = nexttile(t);
    end
    a = normalizeAmp(snaps{k});
    imagesc(ax, [0 xmax], [0 zmax], a);
    axis(ax, 'normal');
    set(ax, 'YDir','reverse', 'FontSize',12, 'LineWidth',0.8, 'Box','on');
    colormap(ax, seismicMap(256));
    caxis(ax, [-0.5 0.5]);
    cb = colorbar(ax);
    cb.Label.String = 'Amplitude';
    cb.Label.FontSize = 12;
    cb.FontSize = 12;
    xlabel(ax, 'Distance (km)', 'FontSize',12);
    ylabel(ax, 'Depth (km)', 'FontSize',12);
    if useManualLayout
        axPos = ax.Position;
        annotation(fig, 'textbox', [axPos(1), axPos(2) - captionGap, axPos(3), 0.026], ...
            'String', sprintf('(%c) t=%.1fs', letters(k), m.times(k)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'LineStyle', 'none', ...
            'FontSize', 12, ...
            'FontWeight', 'bold');
    else
        text(ax, 0.5, -0.18, sprintf('(%c) t=%.1fs', letters(k), m.times(k)), ...
            'Units', 'normalized', ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', ...
            'FontSize', 12, ...
            'FontWeight', 'bold', ...
            'Clipping', 'off');
    end
end
exportgraphics(fig, outPath, 'Resolution', 300);
close(fig);
end

function plotElasticPairSet(vxSnaps, vzSnaps, m, outPath)
ntimes = numel(m.times);
fig = figure('Visible','off','Position',[60 20 900 1850]);
t = tiledlayout(fig, ntimes, 2, 'TileSpacing','compact', 'Padding','compact');
colormap(fig, seismicMap(256));
xmax = (m.nx - 1) * m.dx;
zmax = (m.nz - 1) * m.dz;
lim = pairPercentile(vxSnaps, vzSnaps, 99.5);
if lim <= 0
    lim = 1;
end
for k = 1:ntimes
    for j = 1:2
        ax = nexttile(t);
        if j == 1
            a = normalizeWithLimit(vxSnaps{k}, lim);
            comp = 'v_x';
        else
            a = normalizeWithLimit(vzSnaps{k}, lim);
            comp = 'v_z';
        end
        imagesc(ax, [0 xmax], [0 zmax], a);
        axis(ax, 'image');
        set(ax, 'YDir','reverse', 'FontSize',8, 'LineWidth',0.75, 'Box','on');
        caxis(ax, [-0.5 0.5]);
        title(ax, sprintf('%s, t=%.1fs', comp, m.times(k)), 'FontSize',10, 'FontWeight','bold');
        xlabel(ax, 'Distance (km)', 'FontSize',8);
        ylabel(ax, 'Depth (km)', 'FontSize',8);
    end
end
cb = colorbar(ax);
cb.Layout.Tile = 'east';
cb.Label.String = 'Amplitude';
cb.Label.FontSize = 9;
cb.FontSize = 8;
exportgraphics(fig, outPath, 'Resolution', 220);
close(fig);
end

function plotRecord(rec, m, outPath)
xmax = (m.nx - 1) * m.dx;
tmax = (m.nt - 1) * m.dt;
rec = rec .* linspace(1, 2.5, size(rec,1))';
lim = percentile(abs(rec(:)), 99.2);
if lim <= 0
    lim = 1;
end
fig = figure('Visible','off','Position',[100 100 680 530]);
ax = axes(fig);
imagesc(ax, [0 xmax], [0 tmax], rec);
set(ax, 'YDir','reverse', 'FontSize',9, 'LineWidth',0.9, 'Box','on');
colormap(ax, gray(256));
caxis(ax, [-lim lim]);
xlabel(ax, 'Distance (km)', 'FontSize',10);
ylabel(ax, 'Time (s)', 'FontSize',10);
exportgraphics(fig, outPath, 'Resolution', 300);
close(fig);
end

function a = normalizeAmp(a)
lim = percentile(abs(a(:)), 99.5);
if lim <= 0
    lim = 1;
end
a = max(min(a ./ lim .* 0.5, 0.5), -0.5);
a(abs(a) < 0.012) = 0;
end

function a = normalizeWithLimit(a, lim)
a = max(min(double(a) ./ lim .* 0.5, 0.5), -0.5);
a(abs(a) < 0.012) = 0;
end

function lim = pairPercentile(vxSnaps, vzSnaps, q)
x = [];
for k = 1:numel(vxSnaps)
    x = [x; abs(vxSnaps{k}(:)); abs(vzSnaps{k}(:))]; %#ok<AGROW>
end
lim = percentile(x, q);
end

function data = readBin(path, shape)
fid = fopen(path, 'rb');
if fid < 0
    error('Cannot open %s', path);
end
data = fread(fid, prod(shape), 'single=>single');
fclose(fid);
if numel(data) ~= prod(shape)
    error('Unexpected size for %s: %d vs %d', path, numel(data), prod(shape));
end
data = reshape(data, shape);
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

function suffix = timeSuffix(t)
suffix = sprintf('%0.1f', t);
suffix = strrep(suffix, '.', 'p');
end

function ensureDir(path)
if ~exist(path, 'dir')
    mkdir(path);
end
end

function p = percentile(x, q)
x = sort(x(:));
if isempty(x)
    p = 0;
    return;
end
idx = max(1, min(numel(x), round(q/100 * numel(x))));
p = double(x(idx));
end
