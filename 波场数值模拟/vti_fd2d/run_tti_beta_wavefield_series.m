clear; clc; close all;

outDir = 'D:\ryjin\paper_figures_source\vti\tti_beta_series_vti_settings';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Use the existing VTI program settings, then rotate the same stiffness
% tensor to TTI for beta comparison.
nx = 401;
nz = 401;
dx = 10.0;
dz = 10.0;
dt = 1.0e-3;
snapTime = 0.60;
nt = round(snapTime / dt);
f0 = 10;
t0 = 1.0 / f0;

% Model-1 parameters in vti_fd_chapter_data.m:
% vp0=2590.885..., vs0=1435.166..., rho=2680, epsilon=0.1981656,
% delta=-0.236745, gamma=0.0298913.
c11_0 = 25.12e9;
c13_0 = 1.50e9;
c33_0 = 17.99e9;
c55_0 = 5.52e9;
rho0  = 2680;

sourceAmp = 6.0e8;
xs = round((nx + 1) / 2);
zs = round((nz + 1) / 2);
sourceSigmaGrid = 2.5;

betaList = [-90, -60, -30, 0, 30, 60];
snapVx = cell(numel(betaList), 1);
snapVz = cell(numel(betaList), 1);

for ib = 1:numel(betaList)
    beta = betaList(ib);
    fprintf('Running TTI beta = %+d deg\n', beta);
    [vxSnap, vzSnap] = runOneTTI(beta, nx, nz, dx, dz, dt, nt, f0, t0, ...
        sourceAmp, xs, zs, sourceSigmaGrid, rho0, c11_0, c13_0, c33_0, c55_0);
    snapVx{ib} = vxSnap;
    snapVz{ib} = vzSnap;

    figPath = fullfile(outDir, sprintf('tti_beta_%+d_wavefield_vx_vz.png', beta));
    saveOneWavefield(figPath, vxSnap, vzSnap, beta, dx, dz, snapTime);
end

panelPath = fullfile(outDir, 'tti_beta_wavefield_panel.png');
savePanel(panelPath, snapVx, snapVz, betaList, dx, dz);

fprintf('Finished TTI beta wavefield figures: %s\n', outDir);

function [vx, vz] = runOneTTI(betaDeg, nx, nz, dx, dz, dt, nt, f0, t0, ...
    sourceAmp, xs, zs, sourceSigmaGrid, rho0, c11, c13, c33, c55)

    [C11, C13, C33, C15, C35, C55] = rotateTTI(betaDeg, c11, c13, c33, c55);

    sxx = zeros(nz, nx);
    szz = zeros(nz, nx);
    sxz = zeros(nz, nx);
    vx = zeros(nz, nx);
    vz = zeros(nz, nx);

    damp = makeSponge(nz, nx, 40, 0.022);
    srcMask = makeGaussianSource(nz, nx, zs, xs, sourceSigmaGrid);

    for it = 1:nt
        dsxx_dx = ddx4(sxx, dx);
        dsxz_dz = ddz4(sxz, dz);
        dsxz_dx = ddx4(sxz, dx);
        dszz_dz = ddz4(szz, dz);

        vx = vx + dt / rho0 * (dsxx_dx + dsxz_dz);
        vz = vz + dt / rho0 * (dsxz_dx + dszz_dz);

        src = sourceAmp * ricker((it - 1) * dt, f0, t0);
        sxx = sxx + src * srcMask;
        szz = szz + src * srcMask;

        dvx_dx = ddx4(vx, dx);
        dvz_dz = ddz4(vz, dz);
        dvx_dz = ddz4(vx, dz);
        dvz_dx = ddx4(vz, dx);
        gam_xz = dvx_dz + dvz_dx;

        sxx = sxx + dt * (C11 * dvx_dx + C13 * dvz_dz + C15 * gam_xz);
        szz = szz + dt * (C13 * dvx_dx + C33 * dvz_dz + C35 * gam_xz);
        sxz = sxz + dt * (C15 * dvx_dx + C35 * dvz_dz + C55 * gam_xz);

        vx = vx .* damp;
        vz = vz .* damp;
        sxx = sxx .* damp;
        szz = szz .* damp;
        sxz = sxz .* damp;
    end
end

function mask = makeGaussianSource(nz, nx, zs, xs, sigmaGrid)
    [X, Z] = meshgrid(1:nx, 1:nz);
    mask = exp(-((X - xs).^2 + (Z - zs).^2) / (2 * sigmaGrid^2));
    mask = mask / sum(mask(:));
end

function [C11, C13, C33, C15, C35, C55] = rotateTTI(betaDeg, c11, c13, c33, c55)
    th = deg2rad(betaDeg);
    a = cos(th);
    b = sin(th);
    A = c11 - c13 - 2*c55;
    B = c33 - c13 - 2*c55;

    C11 = c11*a^4 + c33*b^4 + 2*(c13 + 2*c55)*a^2*b^2;
    C33 = c11*b^4 + c33*a^4 + 2*(c13 + 2*c55)*a^2*b^2;
    C13 = (c11 + c33 - 4*c55)*a^2*b^2 + c13*(a^4 + b^4);
    C55 = (c11 + c33 - 2*c13 - 2*c55)*a^2*b^2 + c55*(a^4 + b^4);
    C15 = A*a^3*b - B*a*b^3;
    C35 = A*a*b^3 - B*a^3*b;
end

function d = ddx4(f, h)
    d = zeros(size(f));
    d(:, 3:end-2) = (-f(:, 5:end) + 8*f(:, 4:end-1) - 8*f(:, 2:end-3) + f(:, 1:end-4)) / (12*h);
end

function d = ddz4(f, h)
    d = zeros(size(f));
    d(3:end-2, :) = (-f(5:end, :) + 8*f(4:end-1, :) - 8*f(2:end-3, :) + f(1:end-4, :)) / (12*h);
end

function y = ricker(t, f0, t0)
    a = (pi * f0 * (t - t0)).^2;
    y = (1 - 2*a) .* exp(-a);
end

function damp = makeSponge(nz, nx, nb, strength)
    damp = ones(nz, nx);
    for i = 1:nb
        x = (nb - i + 1) / nb;
        val = exp(-strength * x^2);
        damp(i, :) = damp(i, :) * val;
        damp(nz-i+1, :) = damp(nz-i+1, :) * val;
        damp(:, i) = damp(:, i) * val;
        damp(:, nx-i+1) = damp(:, nx-i+1) * val;
    end
end

function saveOneWavefield(figPath, vx, vz, beta, dx, dz, snapTime)
    x_km = (0:size(vx, 2)-1) * dx / 1000;
    z_km = (0:size(vx, 1)-1) * dz / 1000;
    clipVal = robustClip([vx(:); vz(:)], 0.995);
    cmap = seismicCmap(256);

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 980, 430], ...
        'ToolBar', 'none', 'MenuBar', 'none');
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    imagesc(x_km, z_km, vx);
    axis image; set(gca, 'YDir', 'reverse', 'FontSize', 12);
    colormap(gca, cmap); caxis([-clipVal, clipVal]);
    title('Vx', 'FontSize', 14);
    xlabel('Distance (km)'); ylabel('Depth (km)');
    colorbar;

    nexttile;
    imagesc(x_km, z_km, vz);
    axis image; set(gca, 'YDir', 'reverse', 'FontSize', 12);
    colormap(gca, cmap); caxis([-clipVal, clipVal]);
    title('Vz', 'FontSize', 14);
    xlabel('Distance (km)'); ylabel('Depth (km)');
    colorbar;

    exportgraphics(fig, figPath, 'Resolution', 300);
    close(fig);
end

function savePanel(panelPath, snapVx, snapVz, betaList, dx, dz)
    x_km = (0:size(snapVx{1}, 2)-1) * dx / 1000;
    z_km = (0:size(snapVx{1}, 1)-1) * dz / 1000;
    cmap = seismicCmap(256);

    fig = figure('Color', 'w', 'Position', [50, 50, 1500, 1150]);
    tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

    for ib = 1:numel(betaList)
        clipVal = robustClip([snapVx{ib}(:); snapVz{ib}(:)], 0.995);
        nexttile;
        imagesc(x_km, z_km, snapVx{ib});
        axis image; set(gca, 'YDir', 'reverse', 'FontSize', 10);
        colormap(gca, cmap); caxis([-clipVal, clipVal]);
        title(sprintf('(a) Vx, \\beta=%+d^\\circ', betaList(ib)), 'FontSize', 12);
        xlabel('Distance (km)'); ylabel('Depth (km)');
        colorbar;

        nexttile;
        imagesc(x_km, z_km, snapVz{ib});
        axis image; set(gca, 'YDir', 'reverse', 'FontSize', 10);
        colormap(gca, cmap); caxis([-clipVal, clipVal]);
        title(sprintf('(b) Vz, \\beta=%+d^\\circ', betaList(ib)), 'FontSize', 12);
        xlabel('Distance (km)'); ylabel('Depth (km)');
        colorbar;
    end

    exportgraphics(fig, panelPath, 'Resolution', 300);
    close(fig);
end

function cmap = seismicCmap(m)
    cmap = [linspace(0,1,m/2)', linspace(0,1,m/2)', ones(m/2,1); ...
            ones(m/2,1), linspace(1,0,m/2)', linspace(1,0,m/2)'];
end

function clipVal = robustClip(A, ratio)
    v = sort(abs(A(:)));
    idx = max(1, min(numel(v), round(ratio * numel(v))));
    clipVal = v(idx);
    if clipVal <= 0
        clipVal = max(v);
    end
    if clipVal <= 0
        clipVal = 1;
    end
end
