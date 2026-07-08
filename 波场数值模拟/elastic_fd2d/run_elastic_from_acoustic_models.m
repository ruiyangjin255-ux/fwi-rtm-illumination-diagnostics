function run_elastic_from_acoustic_models()
% Elastic wavefield simulation using explicitly defined elastic models.
% Uses a 4th-order staggered-grid velocity-stress scheme with split PML.

close all; clc;

outDir = 'D:\ryjin\paper_figures_source\elastic';
resDir = 'D:\ryjin\elastic_fd2d\results';
ensureDir(outDir);
ensureDir(resDir);

models = {
    struct('key','uniform_explosion','label','Homogeneous - Explosion Source','type','uniform', ...
        'nx',401,'nz',401, 'nt',1001, 'src',[201 1], 'recDepth',5, ...
        'sourceId',1, 'vp',2000,'vs',1000,'rho',2000), ...
    struct('key','uniform_x_force','label','Homogeneous - Horizontal Force Source','type','uniform', ...
        'nx',401,'nz',401, 'nt',1001, 'src',[201 1], 'recDepth',5, ...
        'sourceId',2, 'vp',2000,'vs',1000,'rho',2000), ...
    struct('key','uniform_z_force','label','Homogeneous - Vertical Force Source','type','uniform', ...
        'nx',401,'nz',401, 'nt',1001, 'src',[201 1], 'recDepth',5, ...
        'sourceId',3, 'vp',2000,'vs',1000,'rho',2000), ...
    struct('key','uniform_shear','label','Homogeneous - Shear Source','type','uniform', ...
        'nx',401,'nz',401, 'nt',1001, 'src',[201 1], 'recDepth',5, ...
        'sourceId',4, 'vp',2000,'vs',1000,'rho',2000)
};

for im = 1:numel(models)
    fprintf('\nRunning elastic model: %s\n', models{im}.key);
    runOneModel(models{im}, outDir, resDir);
end
fprintf('\nFinished elastic figures under %s\n', outDir);
end

function runOneModel(m, outDir, resDir)
dx = 10; dz = 10;
dt = 0.001;
if isfield(m, 'nt')
    nt = m.nt;
else
    nt = 1401;
end
f0 = 10;
t0 = 1 / f0;
nabc = 25;
pmlOrder = 2;
Rcoef = 1e-6;
plotTimes = [];
plotSteps = round(plotTimes / dt) + 1;

nx0 = m.nx; nz0 = m.nz;
nx = nx0 + 2*nabc;
nz = nz0 + 2*nabc;

[vp0, vs0, rho0] = buildElasticModel(m, dx, dz);

vp = padEdge(vp0, nabc);
vs = padEdge(vs0, nabc);
rho = padEdge(rho0, nabc);

xs = min(max(m.src(1), 1), nx0) + nabc;
zs = min(max(m.src(2), 1), nz0) + nabc;
receiverDepth = min(max(m.recDepth, 1), nz0) + nabc;

mu = rho .* vs.^2;
lam = rho .* vp.^2 - 2 .* mu;
vmax = max(vp(:));
cfl = vmax * dt / min(dx, dz);
fprintf('  CFL = %.3f\n', cfl);
if cfl >= 0.55
    error('CFL is too large for %s: %.3f', m.key, cfl);
end

txx_x = zeros(nz, nx);  txx_z = zeros(nz, nx);
tzz_x = zeros(nz, nx);  tzz_z = zeros(nz, nx);
vx_x  = zeros(nz, nx+1); vx_z  = zeros(nz, nx+1);
vz_x  = zeros(nz+1, nx); vz_z  = zeros(nz+1, nx);
txz_x = zeros(nz+1, nx+1); txz_z = zeros(nz+1, nx+1);

seisVx = zeros(nt, nx0);
seisVz = zeros(nt, nx0);
vxSnaps = cell(1, numel(plotSteps));
vzSnaps = cell(1, numel(plotSteps));
snapId = 1;

c1 = 9/8;
c2 = -1/24;

sigmaXC = pmlSigma1d(nx, nabc, dx, vmax, Rcoef, pmlOrder);
sigmaZC = pmlSigma1d(nz, nabc, dz, vmax, Rcoef, pmlOrder).';
sigmaXH = zeros(1, nx+1);
sigmaXH(2:nx) = 0.5 * (sigmaXC(1:nx-1) + sigmaXC(2:nx));
sigmaXH(1) = sigmaXC(1); sigmaXH(nx+1) = sigmaXC(nx);
sigmaZH = zeros(nz+1, 1);
sigmaZH(2:nz) = 0.5 * (sigmaZC(1:nz-1) + sigmaZC(2:nz));
sigmaZH(1) = sigmaZC(1); sigmaZH(nz+1) = sigmaZC(nz);

[aXC, bXC] = pmlAB(repmat(sigmaXC, nz, 1), dt);
[aZC, bZC] = pmlAB(repmat(sigmaZC, 1, nx), dt);
[aXVx, bXVx] = pmlAB(repmat(sigmaXH, nz, 1), dt);
[aZVx, bZVx] = pmlAB(repmat(sigmaZC, 1, nx+1), dt);
[aXVz, bXVz] = pmlAB(repmat(sigmaXC, nz+1, 1), dt);
[aZVz, bZVz] = pmlAB(repmat(sigmaZH, 1, nx), dt);
[aXTxz, bXTxz] = pmlAB(repmat(sigmaXH, nz+1, 1), dt);
[aZTxz, bZTxz] = pmlAB(repmat(sigmaZH, 1, nx+1), dt);

tvec = (0:nt-1) * dt;
arg = (pi * f0 * (tvec - t0)).^2;
wavelet = (1 - 2 * arg) .* exp(-arg);
srcScale = 1.0;
sourceId = 1;
if isfield(m, 'sourceId')
    sourceId = m.sourceId;
end

for it = 1:nt
    txx = txx_x + txx_z;
    tzz = tzz_x + tzz_z;
    txz = txz_x + txz_z;

    jv = 3:nz-2; iv = 3:nx-2;
    dtxxDx = (c1 * (txx(jv,iv) - txx(jv,iv-1)) + c2 * (txx(jv,iv+1) - txx(jv,iv-2))) / dx;
    dtxzDz = (c1 * (txz(jv+1,iv) - txz(jv,iv)) + c2 * (txz(jv+2,iv) - txz(jv-1,iv))) / dz;
    rhoVx = 0.5 * (rho(jv,iv) + rho(jv,iv-1));
    vx_x(jv,iv) = bXVx(jv,iv) .* vx_x(jv,iv) + aXVx(jv,iv) .* (dtxxDx ./ rhoVx);
    vx_z(jv,iv) = bZVx(jv,iv) .* vx_z(jv,iv) + aZVx(jv,iv) .* (dtxzDz ./ rhoVx);

    dtxzDx = (c1 * (txz(jv,iv+1) - txz(jv,iv)) + c2 * (txz(jv,iv+2) - txz(jv,iv-1))) / dx;
    dtzzDz = (c1 * (tzz(jv,iv) - tzz(jv-1,iv)) + c2 * (tzz(jv+1,iv) - tzz(jv-2,iv))) / dz;
    rhoVz = 0.5 * (rho(jv,iv) + rho(jv-1,iv));
    vz_x(jv,iv) = bXVz(jv,iv) .* vz_x(jv,iv) + aXVz(jv,iv) .* (dtxzDx ./ rhoVz);
    vz_z(jv,iv) = bZVz(jv,iv) .* vz_z(jv,iv) + aZVz(jv,iv) .* (dtzzDz ./ rhoVz);

    vx = vx_x + vx_z;
    vz = vz_x + vz_z;

    if it <= round(2*t0/dt)
        src = srcScale * wavelet(it);
        switch sourceId
            case 1
                txx_x(zs,xs) = txx_x(zs,xs) + 0.5 * src;
                txx_z(zs,xs) = txx_z(zs,xs) + 0.5 * src;
                tzz_x(zs,xs) = tzz_x(zs,xs) + 0.5 * src;
                tzz_z(zs,xs) = tzz_z(zs,xs) + 0.5 * src;
            case 2
                vx_x(zs,xs) = vx_x(zs,xs) + 0.5 * src;
                vx_z(zs,xs) = vx_z(zs,xs) + 0.5 * src;
            case 3
                vz_x(zs,xs) = vz_x(zs,xs) + 0.5 * src;
                vz_z(zs,xs) = vz_z(zs,xs) + 0.5 * src;
            case 4
                srcHalf = 3;
                [kx, kz] = meshgrid(-srcHalf:srcHalf, -srcHalf:srcHalf);
                srcKernel = exp(-(kx.^2 + kz.^2) / (2 * 1.15^2));
                srcKernel = srcKernel / sum(srcKernel(:));
                srcJ = (zs-srcHalf):(zs+srcHalf);
                srcI = (xs-srcHalf):(xs+srcHalf);
                srcAmp = src * srcKernel;
                txz_x(srcJ, srcI) = txz_x(srcJ, srcI) + 0.5 * srcAmp;
                txz_z(srcJ, srcI) = txz_z(srcJ, srcI) + 0.5 * srcAmp;
            otherwise
                error('Unsupported source type: %d', sourceId);
        end
        vx = vx_x + vx_z;
        vz = vz_x + vz_z;
    end

    js = 3:nz-2; is = 3:nx-2;
    dvxDx = (c1 * (vx(js,is+1) - vx(js,is)) + c2 * (vx(js,is+2) - vx(js,is-1))) / dx;
    dvzDz = (c1 * (vz(js+1,is) - vz(js,is)) + c2 * (vz(js+2,is) - vz(js-1,is))) / dz;
    txx_x(js,is) = bXC(js,is) .* txx_x(js,is) + aXC(js,is) .* ((lam(js,is) + 2*mu(js,is)) .* dvxDx);
    txx_z(js,is) = bZC(js,is) .* txx_z(js,is) + aZC(js,is) .* (lam(js,is) .* dvzDz);
    tzz_x(js,is) = bXC(js,is) .* tzz_x(js,is) + aXC(js,is) .* (lam(js,is) .* dvxDx);
    tzz_z(js,is) = bZC(js,is) .* tzz_z(js,is) + aZC(js,is) .* ((lam(js,is) + 2*mu(js,is)) .* dvzDz);

    dvxDz = (c1 * (vx(js,is) - vx(js-1,is)) + c2 * (vx(js+1,is) - vx(js-2,is))) / dz;
    dvzDx = (c1 * (vz(js,is) - vz(js,is-1)) + c2 * (vz(js,is+1) - vz(js,is-2))) / dx;
    muTxz = 0.25 * (mu(js,is) + mu(js-1,is) + mu(js,is-1) + mu(js-1,is-1));
    txz_x(js,is) = bXTxz(js,is) .* txz_x(js,is) + aXTxz(js,is) .* (muTxz .* dvzDx);
    txz_z(js,is) = bZTxz(js,is) .* txz_z(js,is) + aZTxz(js,is) .* (muTxz .* dvxDz);

    vx = vx_x + vx_z;
    vz = vz_x + vz_z;
    vxC = 0.5 * (vx(:,1:nx) + vx(:,2:nx+1));
    vzC = 0.5 * (vz(1:nz,:) + vz(2:nz+1,:));
    seisVx(it,:) = vxC(receiverDepth, nabc+1:nabc+nx0);
    seisVz(it,:) = vzC(receiverDepth, nabc+1:nabc+nx0);

    if snapId <= numel(plotSteps) && it == plotSteps(snapId)
        vxSnaps{snapId} = vxC(nabc+1:end-nabc, nabc+1:end-nabc);
        vzSnaps{snapId} = vzC(nabc+1:end-nabc, nabc+1:end-nabc);
        snapId = snapId + 1;
    end
    if mod(it, 200) == 0
        fprintf('  it = %d / %d\n', it, nt);
    end
end

xKm = (0:nx0-1) * dx / 1000;
zKm = (0:nz0-1) * dz / 1000;
tAxis = (0:nt-1) * dt;

writeFloatBin(fullfile(resDir, [m.key '_vp.bin']), single(vp0));
writeFloatBin(fullfile(resDir, [m.key '_vs.bin']), single(vs0));
writeFloatBin(fullfile(resDir, [m.key '_record_vx.bin']), single(seisVx));
writeFloatBin(fullfile(resDir, [m.key '_record_vz.bin']), single(seisVz));
for k = 1:numel(plotTimes)
    suffix = timeSuffix(plotTimes(k));
    writeFloatBin(fullfile(resDir, sprintf('%s_vx_%ss.bin', m.key, suffix)), single(vxSnaps{k}));
    writeFloatBin(fullfile(resDir, sprintf('%s_vz_%ss.bin', m.key, suffix)), single(vzSnaps{k}));
    plotSnapshotPair(vxSnaps{k}, vzSnaps{k}, xKm, zKm, m, plotTimes(k), ...
        fullfile(outDir, sprintf('%s_wavefield_%ss_vx_vz.png', m.key, suffix)));
end
plotShotGatherPair(seisVx, seisVz, xKm, tAxis, m, ...
    fullfile(outDir, sprintf('%s_record_vx_vz.png', m.key)));

if isfield(m, 'snapSrc')
    mSnap = m;
    mSnap.src = m.snapSrc;
    mSnap.key = [m.key '_snap'];
    mSnap.nt = round(max(plotTimes) / dt) + 1;
    mSnap = rmfield(mSnap, 'snapSrc');
    fprintf('  rerunning %s wavefield snapshots with source at %.0f m, %.0f m\n', ...
        m.key, (mSnap.src(1)-1)*dx, (mSnap.src(2)-1)*dz);
    runOneModel(mSnap, outDir, resDir);
    for k = 1:numel(plotTimes)
        suffix = timeSuffix(plotTimes(k));
        copyfile(fullfile(outDir, sprintf('%s_wavefield_%ss_vx_vz.png', mSnap.key, suffix)), ...
            fullfile(outDir, sprintf('%s_wavefield_%ss_vx_vz.png', m.key, suffix)), 'f');
        copyfile(fullfile(resDir, sprintf('%s_vx_%ss.bin', mSnap.key, suffix)), ...
            fullfile(resDir, sprintf('%s_vx_%ss.bin', m.key, suffix)), 'f');
        copyfile(fullfile(resDir, sprintf('%s_vz_%ss.bin', mSnap.key, suffix)), ...
            fullfile(resDir, sprintf('%s_vz_%ss.bin', m.key, suffix)), 'f');
    end
    delete(fullfile(outDir, [mSnap.key '_*.png']));
    delete(fullfile(resDir, [mSnap.key '_*.bin']));
end
end

function [vp, vs, rho] = buildElasticModel(m, dx, dz)
vp = zeros(m.nz, m.nx, 'single');
vs = zeros(m.nz, m.nx, 'single');
rho = zeros(m.nz, m.nx, 'single');

switch lower(m.type)
    case 'uniform'
        vp(:) = m.vp;
        vs(:) = m.vs;
        rho(:) = m.rho;
    case 'layer'
        interfaceRow = floor(m.interfaceDepth / dz) + 1;
        upper = (1:m.nz)' < interfaceRow;
        vp(upper,:) = m.vpUpper;
        vs(upper,:) = m.vsUpper;
        rho(upper,:) = m.rhoUpper;
        vp(~upper,:) = m.vpLower;
        vs(~upper,:) = m.vsLower;
        rho(~upper,:) = m.rhoLower;
    otherwise
        error('Unsupported elastic model type: %s', m.type);
end
end

function plotSnapshotPair(vx, vz, xKm, zKm, m, timeValue, outPath)
cmap = seismicMap(256);
lim = percentile(abs([vx(:); vz(:)]), 99.5);
if lim <= 0, lim = 1; end
lim = 0.55 * lim;
fig = figure('Visible','off','Color','w','Position',[100 100 1100 450]);
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

function plotShotGatherPair(seisVx, seisVz, xKm, tAxis, m, outPath)
seisVx = seisVx ./ max(max(abs(seisVx)), eps);
seisVz = seisVz ./ max(max(abs(seisVz)), eps);
fig = figure('Visible','off','Color','w','Position',[100 100 1000 450]);
ax = subplot(1,2,1);
imagesc(ax, xKm, tAxis, seisVx); set(ax,'YDir','reverse'); colormap(ax, gray(256));
applyTimeTicks(ax, tAxis);
clim(ax, [-0.08 0.08]); title(ax, 'Shot Gather - Vx', 'FontSize',13);
xlabel(ax, 'Distance (km)'); ylabel(ax, 'Time (s)');
ax = subplot(1,2,2);
imagesc(ax, xKm, tAxis, seisVz); set(ax,'YDir','reverse'); colormap(ax, gray(256));
applyTimeTicks(ax, tAxis);
clim(ax, [-0.08 0.08]); title(ax, 'Shot Gather - Vz', 'FontSize',13);
xlabel(ax, 'Distance (km)'); ylabel(ax, 'Time (s)');
sgtitle(fig, 'Shot Gathers - Homogeneous (PML)', 'FontSize', 14, 'FontWeight', 'bold');
exportgraphics(fig, outPath, 'Resolution', 300);
close(fig);
end

function applyTimeTicks(ax, tAxis)
yt = 0:0.1:max(tAxis);
yticks(ax, yt);
ytLabel = strings(size(yt));
for ii = 1:length(yt)
    if abs(mod(yt(ii), 0.5)) < 1e-8 || abs(mod(yt(ii), 0.5) - 0.5) < 1e-8
        ytLabel(ii) = num2str(yt(ii), '%.1f');
    else
        ytLabel(ii) = "";
    end
end
yticklabels(ax, ytLabel);
end

function plotElasticModel(vp, vs, xKm, zKm, m, outPath)
fig = figure('Visible','off','Color','w','Position',[100 100 620 560]);
ax = axes(fig);
imagesc(ax, xKm, zKm, vp/1000);
set(ax,'YDir','reverse');
axis(ax,'image');
colormap(ax, jet(256));
cb = colorbar(ax);
cb.Label.String = 'V_p (km/s)';
title(ax, sprintf('Elastic Model - %s', m.label), 'FontSize',14, 'FontWeight','bold');
xlabel(ax, 'Distance (km)');
ylabel(ax, 'Depth (km)');
hold(ax, 'on');
srcXKm = xKm(m.src(1));
srcZKm = zKm(m.src(2));
plot(ax, srcXKm, srcZKm, ...
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

function writeFloatBin(path, a)
fid = fopen(path, 'wb');
if fid < 0, error('Cannot write %s', path); end
fwrite(fid, a, 'single');
fclose(fid);
end

function b = padEdge(a, n)
[nz, nx] = size(a);
b = zeros(nz + 2*n, nx + 2*n, 'like', a);
b(n+1:n+nz, n+1:n+nx) = a;
b(1:n, n+1:n+nx) = repmat(a(1,:), n, 1);
b(n+nz+1:end, n+1:n+nx) = repmat(a(end,:), n, 1);
b(:, 1:n) = repmat(b(:, n+1), 1, n);
b(:, n+nx+1:end) = repmat(b(:, n+nx), 1, n);
end

function sigma = pmlSigma1d(n, npml, d, vmax, rcoef, order)
sigma = zeros(1, n);
sigmaMax = - (order + 1) * vmax * log(rcoef) / (2 * npml * d);
for i = 1:n
    if i <= npml
        r = (npml - i + 1) / npml;
        sigma(i) = sigmaMax * r^order;
    elseif i >= n - npml + 1
        r = (i - (n - npml)) / npml;
        sigma(i) = sigmaMax * r^order;
    end
end
end

function [a, b] = pmlAB(sigma, dt)
b = exp(-sigma * dt);
a = zeros(size(sigma));
mask = sigma > 1e-12;
a(mask) = (1 - b(mask)) ./ sigma(mask);
a(~mask) = dt;
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

function ensureDir(path)
if ~exist(path, 'dir'), mkdir(path); end
end
