if ~exist('elasticFD_batch_mode', 'var') || ~elasticFD_batch_mode
    clear; clc; close all;
else
    clc; close all;
end

%% ===================== 1. 参数 =====================
model_id  = 2;   % 1: 均匀介质, 2: 双层介质
source_id = 1;   % 1: 爆炸源, 2: 水平力源, 3: 垂直力源, 4: 剪切源

nx_orig = 401;
if exist('elasticFD_model_id_override', 'var')
    model_id = elasticFD_model_id_override;
end
if exist('elasticFD_source_id_override', 'var')
    source_id = elasticFD_source_id_override;
end
nz_orig = 401;
dx = 10;
dz = 10;              % m

nt = 2000;
if exist('elasticFD_nt_override', 'var')
    nt = elasticFD_nt_override;
end
dt = 0.001;           % s
f0 = 10;
t0 = 1 / f0;

nabc = 25;            % PML 厚度
pml_order = 2;        % PML 阶数
Rcoef = 1e-6;         % 目标反射系数

nx = nx_orig + 2*nabc;
nz = nz_orig + 2*nabc;

% 炮点（原模型坐标）
zs_orig = 1;         % 建议不要太浅，减小顶部PML对近源波场影响
xs_orig = 201;
zs = zs_orig + nabc;
xs = xs_orig + nabc;

receiver_depth = nabc + 5;  % 检波器深度（在扩展模型中）

% 指定输出时刻
plot_times = 0.7;
if exist('elasticFD_plot_times_override', 'var')
    plot_times = elasticFD_plot_times_override;
end
plot_steps = round(plot_times / dt);

%% ===================== 2. 介质模型 =====================
vp  = 2000 * ones(nz, nx);
vs  = 1000 * ones(nz, nx);
rho = 2000 * ones(nz, nx);

[X, Z] = meshgrid(1:nx, 1:nz);

switch model_id
    case 1
        model_name = 'Homogeneous';
        vp(:)  = 2000;
        vs(:)  = 1000;
        rho(:) = 2000;

    case 2
        model_name = 'Two-layer';
        idx = Z > (100 + nabc);
        vp(idx)  = 3000;
        vs(idx)  = 1500;
        rho(idx) = 2200;

    case 3
        model_name = 'Graben';
        vp(:)  = 2200;
        vs(:)  = 1100;
        rho(:) = 2100;

        x_m = (X - nabc - 1) * dx;
        z_m = (Z - nabc - 1) * dz;
        interface_m = 1500 * ones(size(X));

        leftRamp = x_m > 1200 & x_m < 1600;
        interface_m(leftRamp) = 1500 + (x_m(leftRamp) - 1200) / 400 * 500;

        centerBlock = x_m >= 1600 & x_m <= 2400;
        interface_m(centerBlock) = 2000;

        rightRamp = x_m > 2400 & x_m < 2800;
        interface_m(rightRamp) = 2000 - (x_m(rightRamp) - 2400) / 400 * 500;

        idx = z_m >= interface_m;
        vp(idx)  = 3500;
        vs(idx)  = 1900;
        rho(idx) = 2400;
end

mu  = rho .* vs.^2;
lam = rho .* vp.^2 - 2 .* mu;

%% ===================== 3. 稳定性检查 =====================
vmax = max(vp(:));
cfl  = vmax * dt / min(dx, dz);
fprintf('CFL = %.3f\n', cfl);
if cfl >= 0.5
    error('CFL 不满足，当前 CFL = %.3f，请减小 dt 或增大 dx/dz。', cfl);
end

%% ===================== 4. 交错网格变量（PML 分裂场） =====================
% 总场 = x方向分裂项 + z方向分裂项

txx_x = zeros(nz, nx);  txx_z = zeros(nz, nx);
tzz_x = zeros(nz, nx);  tzz_z = zeros(nz, nx);

vx_x  = zeros(nz, nx+1);  vx_z  = zeros(nz, nx+1);
vz_x  = zeros(nz+1, nx);  vz_z  = zeros(nz+1, nx);

txz_x = zeros(nz+1, nx+1); txz_z = zeros(nz+1, nx+1);

% 单炮记录
seis_vx = zeros(nt, nx_orig);
seis_vz = zeros(nt, nx_orig);

% 快照
vx_snaps = cell(1, numel(plot_steps));
vz_snaps = cell(1, numel(plot_steps));
sid = 1;

%% ===================== 5. 四阶差分系数 =====================
% 4th-order staggered FD coefficients
c1 = 9/8;
c2 = -1/24;

%% ===================== 6. PML 系数 =====================
% 中心网格与半格点网格的一维阻尼剖面
sigma_x_c_1d = pml_sigma_1d(nx, nabc, dx, vmax, Rcoef, pml_order);
sigma_z_c_1d = pml_sigma_1d(nz, nabc, dz, vmax, Rcoef, pml_order).';

sigma_x_h_1d = zeros(1, nx+1);
sigma_x_h_1d(2:nx) = 0.5 * (sigma_x_c_1d(1:nx-1) + sigma_x_c_1d(2:nx));
sigma_x_h_1d(1)    = sigma_x_c_1d(1);
sigma_x_h_1d(nx+1) = sigma_x_c_1d(nx);

sigma_z_h_1d = zeros(nz+1, 1);
sigma_z_h_1d(2:nz) = 0.5 * (sigma_z_c_1d(1:nz-1) + sigma_z_c_1d(2:nz));
sigma_z_h_1d(1)    = sigma_z_c_1d(1);
sigma_z_h_1d(nz+1) = sigma_z_c_1d(nz);

% 展开为二维，与各交错网格对齐
sigma_x_c   = repmat(sigma_x_c_1d, nz, 1);
sigma_z_c   = repmat(sigma_z_c_1d, 1, nx);

sigma_x_vx  = repmat(sigma_x_h_1d, nz, 1);
sigma_z_vx  = repmat(sigma_z_c_1d, 1, nx+1);

sigma_x_vz  = repmat(sigma_x_c_1d, nz+1, 1);
sigma_z_vz  = repmat(sigma_z_h_1d, 1, nx);

sigma_x_txz = repmat(sigma_x_h_1d, nz+1, 1);
sigma_z_txz = repmat(sigma_z_h_1d, 1, nx+1);

% 指数积分更新系数：u^{n+1} = b*u^n + a*rhs
[a_x_c,   b_x_c]   = pml_ab(sigma_x_c, dt);
[a_z_c,   b_z_c]   = pml_ab(sigma_z_c, dt);
[a_x_vx,  b_x_vx]  = pml_ab(sigma_x_vx, dt);
[a_z_vx,  b_z_vx]  = pml_ab(sigma_z_vx, dt);
[a_x_vz,  b_x_vz]  = pml_ab(sigma_x_vz, dt);
[a_z_vz,  b_z_vz]  = pml_ab(sigma_z_vz, dt);
[a_x_txz, b_x_txz] = pml_ab(sigma_x_txz, dt);
[a_z_txz, b_z_txz] = pml_ab(sigma_z_txz, dt);

%% ===================== 7. 震源 =====================
t_vec = (0:nt-1) * dt;
arg = (pi * f0 * (t_vec - t0)).^2;
wavelet = (1 - 2 * arg) .* exp(-arg);

%% ===================== 8. 时间推进 =====================
for it = 1:nt

    % 当前总应力场
    txx = txx_x + txx_z;
    tzz = tzz_x + tzz_z;
    txz = txz_x + txz_z;

    % ---------- 8.1 更新速度 vx ----------
    jv = 3:nz-2;
    iv = 3:nx-2;

    dtxx_dx = ( c1 * (txx(jv, iv)   - txx(jv, iv-1)) + ...
                c2 * (txx(jv, iv+1) - txx(jv, iv-2)) ) / dx;
    dtxz_dz = ( c1 * (txz(jv+1, iv) - txz(jv, iv)) + ...
                c2 * (txz(jv+2, iv) - txz(jv-1, iv)) ) / dz;

    rho_vx = 0.5 * (rho(jv, iv) + rho(jv, iv-1));

    vx_x(jv, iv) = b_x_vx(jv, iv) .* vx_x(jv, iv) + ...
                   a_x_vx(jv, iv) .* (dtxx_dx ./ rho_vx);
    vx_z(jv, iv) = b_z_vx(jv, iv) .* vx_z(jv, iv) + ...
                   a_z_vx(jv, iv) .* (dtxz_dz ./ rho_vx);

    % ---------- 8.2 更新速度 vz ----------
    jv2 = 3:nz-2;
    iv2 = 3:nx-2;

    dtxz_dx = ( c1 * (txz(jv2, iv2+1) - txz(jv2, iv2)) + ...
                c2 * (txz(jv2, iv2+2) - txz(jv2, iv2-1)) ) / dx;
    dtzz_dz = ( c1 * (tzz(jv2, iv2)   - tzz(jv2-1, iv2)) + ...
                c2 * (tzz(jv2+1, iv2) - tzz(jv2-2, iv2)) ) / dz;

    rho_vz = 0.5 * (rho(jv2, iv2) + rho(jv2-1, iv2));

    vz_x(jv2, iv2) = b_x_vz(jv2, iv2) .* vz_x(jv2, iv2) + ...
                     a_x_vz(jv2, iv2) .* (dtxz_dx ./ rho_vz);
    vz_z(jv2, iv2) = b_z_vz(jv2, iv2) .* vz_z(jv2, iv2) + ...
                     a_z_vz(jv2, iv2) .* (dtzz_dz ./ rho_vz);

    % 当前总速度场（用于应力更新）
    vx = vx_x + vx_z;
    vz = vz_x + vz_z;

    % ---------- 8.3 加震源 ----------
    if it <= round(2*t0/dt)
        switch source_id
            case 1 % 爆炸源
                txx_x(zs, xs) = txx_x(zs, xs) + 0.5 * wavelet(it);
                txx_z(zs, xs) = txx_z(zs, xs) + 0.5 * wavelet(it);
                tzz_x(zs, xs) = tzz_x(zs, xs) + 0.5 * wavelet(it);
                tzz_z(zs, xs) = tzz_z(zs, xs) + 0.5 * wavelet(it);

            case 2 % 水平力源（修改：在 vx_x 与 vx_z 上对半注入）
                vx_x(zs, xs) = vx_x(zs, xs) + 0.5 * wavelet(it);
                vx_z(zs, xs) = vx_z(zs, xs) + 0.5 * wavelet(it);

            case 3 % 垂直力源（修改：在 vz_x 与 vz_z 上对半注入）
                vz_x(zs, xs) = vz_x(zs, xs) + 0.5 * wavelet(it);
                vz_z(zs, xs) = vz_z(zs, xs) + 0.5 * wavelet(it);

            case 4 % 剪切力源
                src_half = 9;
                sigma_src = 3.0;
                [kx, kz] = meshgrid(-src_half:src_half, -src_half:src_half);
                src_kernel = exp(-(kx.^2 + kz.^2) / (2 * sigma_src^2));

                % Divergence-free curl source:
                %   vx += dG/dz, vz += -dG/dx
                % This suppresses the compressional part that a txz
                % double-couple source radiates in the shot gather.
                src_vx = -(kz / sigma_src^2) .* src_kernel;
                src_vz =  (kx / sigma_src^2) .* src_kernel;
                norm_val = max([abs(src_vx(:)); abs(src_vz(:))]);
                if norm_val == 0, norm_val = 1; end
                src_vx = src_vx / norm_val;
                src_vz = src_vz / norm_val;

                src_j = (zs-src_half):(zs+src_half);
                src_i = (xs-src_half):(xs+src_half);
                src_scale = 0.28 * wavelet(it);

                vx_z(src_j, src_i) = vx_z(src_j, src_i) + src_scale * src_vx;
                vz_x(src_j, src_i) = vz_x(src_j, src_i) + src_scale * src_vz;

            case 5 % equal-energy x/z force source
                amp = wavelet(it) / sqrt(2);
                vx_x(zs, xs) = vx_x(zs, xs) + 0.5 * amp;
                vx_z(zs, xs) = vx_z(zs, xs) + 0.5 * amp;
                vz_x(zs, xs) = vz_x(zs, xs) + 0.5 * amp;
                vz_z(zs, xs) = vz_z(zs, xs) + 0.5 * amp;
        end
        vx = vx_x + vx_z;
        vz = vz_x + vz_z;
    end

    % ---------- 8.4 更新 txx, tzz ----------
    js = 3:nz-2;
    is = 3:nx-2;

    dvx_dx = ( c1 * (vx(js, is+1) - vx(js, is)) + ...
               c2 * (vx(js, is+2) - vx(js, is-1)) ) / dx;
    dvz_dz = ( c1 * (vz(js+1, is) - vz(js, is)) + ...
               c2 * (vz(js+2, is) - vz(js-1, is)) ) / dz;

    txx_x(js, is) = b_x_c(js, is) .* txx_x(js, is) + ...
                    a_x_c(js, is) .* ((lam(js,is) + 2*mu(js,is)) .* dvx_dx);
    txx_z(js, is) = b_z_c(js, is) .* txx_z(js, is) + ...
                    a_z_c(js, is) .* (lam(js,is) .* dvz_dz);

    tzz_x(js, is) = b_x_c(js, is) .* tzz_x(js, is) + ...
                    a_x_c(js, is) .* (lam(js,is) .* dvx_dx);
    tzz_z(js, is) = b_z_c(js, is) .* tzz_z(js, is) + ...
                    a_z_c(js, is) .* ((lam(js,is) + 2*mu(js,is)) .* dvz_dz);

    % ---------- 8.5 更新 txz ----------
    jt  = 3:nz-2;
    itx = 3:nx-2;

    dvx_dz = ( c1 * (vx(jt, itx)   - vx(jt-1, itx)) + ...
               c2 * (vx(jt+1, itx) - vx(jt-2, itx)) ) / dz;
    dvz_dx = ( c1 * (vz(jt, itx)   - vz(jt, itx-1)) + ...
               c2 * (vz(jt, itx+1) - vz(jt, itx-2)) ) / dx;

    mu_txz = 0.25 * (mu(jt,itx) + mu(jt-1,itx) + mu(jt,itx-1) + mu(jt-1,itx-1));

    txz_x(jt, itx) = b_x_txz(jt, itx) .* txz_x(jt, itx) + ...
                     a_x_txz(jt, itx) .* (mu_txz .* dvz_dx);
    txz_z(jt, itx) = b_z_txz(jt, itx) .* txz_z(jt, itx) + ...
                     a_z_txz(jt, itx) .* (mu_txz .* dvx_dz);

    % ---------- 8.6 记录 ----------
    vx = vx_x + vx_z;
    vz = vz_x + vz_z;

    vx_c = 0.5 * (vx(:,1:nx) + vx(:,2:nx+1));
    vz_c = 0.5 * (vz(1:nz,:) + vz(2:nz+1,:));

    seis_vx(it,:) = vx_c(receiver_depth, nabc+1:nabc+nx_orig);
    seis_vz(it,:) = vz_c(receiver_depth, nabc+1:nabc+nx_orig);

    % ---------- 8.7 保存快照 ----------
    if sid <= numel(plot_steps) && it == plot_steps(sid)
        vx_snaps{sid} = vx_c(nabc+1:end-nabc, nabc+1:end-nabc);
        vz_snaps{sid} = vz_c(nabc+1:end-nabc, nabc+1:end-nabc);
        sid = sid + 1;
    end

    % ---------- 8.8 进度显示 ----------
    if mod(it, 100) == 0
        fprintf('it = %d / %d\n', it, nt);
    end
end

%% ===================== 9. 坐标 =====================
x_km = (0:nx_orig-1) * dx / 1000;
z_km = (0:nz_orig-1) * dz / 1000;
t_axis = (0:nt-1) * dt;

% 红白蓝色标
m = 64;
seismic_cmap = [linspace(0,1,m/2)', linspace(0,1,m/2)', ones(m/2,1); ...
                ones(m/2,1), linspace(1,0,m/2)', linspace(1,0,m/2)'];

%% ===================== 10. 波场快照 =====================
if exist('elasticFD_source_effect_mode', 'var') && elasticFD_source_effect_mode
    if ~exist('elasticFD_source_effect_out_dir', 'var')
        elasticFD_source_effect_out_dir = fullfile(pwd, 'source_effect_outputs');
    end
    if ~exist(elasticFD_source_effect_out_dir, 'dir')
        mkdir(elasticFD_source_effect_out_dir);
    end

    source_tags = {'explosion', 'x_force', 'z_force', 'shear'};
    tag = source_tags{min(source_id, numel(source_tags))};
    vx_show = vx_snaps{end};
    vz_show = vz_snaps{end};
    if source_id == 4
        [nz_show, nx_show] = size(vx_show);
        kx = [0:floor(nx_show/2), -ceil(nx_show/2)+1:-1];
        kz = [0:floor(nz_show/2), -ceil(nz_show/2)+1:-1];
        [KX, KZ] = meshgrid(kx, kz);
        denom = KX.^2 + KZ.^2;
        denom(denom == 0) = 1;
        VXK = fft2(vx_show);
        VZK = fft2(vz_show);
        kdotv = KX .* VXK + KZ .* VZK;
        vx_show = real(ifft2(VXK - KX .* kdotv ./ denom));
        vz_show = real(ifft2(VZK - KZ .* kdotv ./ denom));
    end
    max_val = max([max(abs(vx_show(:))), max(abs(vz_show(:)))]);
    if max_val == 0, max_val = 1; end
    clip_val = 0.5 * max_val;

    fig = figure('Color','w', 'Position', [100, 100, 1100, 450]);

    subplot(1, 2, 1);
    imagesc(x_km, z_km, vx_show);
    set(gca, 'YDir', 'reverse');
    axis image;
    colormap(gca, seismic_cmap);
    clim([-clip_val, clip_val]);
    title('Vx', 'FontSize', 12, 'FontWeight', 'normal');
    xlabel('Distance (km)');
    ylabel('Depth (km)');
    colorbar;

    subplot(1, 2, 2);
    imagesc(x_km, z_km, vz_show);
    set(gca, 'YDir', 'reverse');
    axis image;
    colormap(gca, seismic_cmap);
    clim([-clip_val, clip_val]);
    title('Vz', 'FontSize', 12, 'FontWeight', 'normal');
    xlabel('Distance (km)');
    ylabel('Depth (km)');
    colorbar;

    exportgraphics(fig, fullfile(elasticFD_source_effect_out_dir, ...
        sprintf('source_effect_%02d_%s_VX_VZ.png', source_id, tag)), 'Resolution', 300);
    fprintf('Saved source effect figure: %s\n', elasticFD_source_effect_out_dir);
    return;
end

figure('Color','w', 'Position', [100, 100, 1100, 450]);
for k = 1:length(plot_times)
    max_val = max([max(abs(vx_snaps{k}(:))), max(abs(vz_snaps{k}(:)))]);
    if max_val == 0, max_val = 1; end

    subplot(1, 2, 1);
    imagesc(x_km, z_km, vx_snaps{k});
    set(gca, 'YDir', 'reverse');
    axis image;
    colormap(gca, seismic_cmap);
    clim([-0.5*max_val, 0.5*max_val]);
    title(['Vx at t=', num2str(plot_times(k),'%.1f'), 's']);
    xlabel('Distance (km)'); ylabel('Depth (km)');
    colorbar;

    subplot(1, 2, 2);
    imagesc(x_km, z_km, vz_snaps{k});
    set(gca, 'YDir', 'reverse');
    axis image;
    colormap(gca, seismic_cmap);
    clim([-0.5*max_val, 0.5*max_val]);
    title(['Vz at t=', num2str(plot_times(k),'%.1f'), 's']);
    xlabel('Distance (km)'); ylabel('Depth (km)');
    colorbar;
end
sgtitle(['Wavefield Snapshots - ', model_name, ' (PML)'], 'FontSize', 14, 'FontWeight', 'bold');

%% ===================== 11. 地震单炮记录 =====================
% 修改：改为全局归一化，不再逐道归一化
amp_vx = max(abs(seis_vx(:)));
amp_vz = max(abs(seis_vz(:)));
if amp_vx > 0
    seis_vx = seis_vx / amp_vx;
end
if amp_vz > 0
    seis_vz = seis_vz / amp_vz;
end

figure('Color','w', 'Position', [100, 100, 1000, 450]);

subplot(1,2,1);
imagesc(x_km, t_axis, seis_vx);
colormap(gray);
set(gca,'YDir','reverse');
apply_time_ticks(t_axis);
amp = max(abs(seis_vx(:))); if amp == 0, amp = 1; end
clim([-0.08*amp, 0.08*amp]);
xlabel('Distance (km)');
ylabel('Time (s)');
title('Shot Gather - Vx');

subplot(1,2,2);
imagesc(x_km, t_axis, seis_vz);
colormap(gray);
set(gca,'YDir','reverse');
apply_time_ticks(t_axis);
amp = max(abs(seis_vz(:))); if amp == 0, amp = 1; end
clim([-0.08*amp, 0.08*amp]);
xlabel('Distance (km)');
ylabel('Time (s)');
title('Shot Gather - Vz');

sgtitle(['Shot Gathers - ', model_name, ' (PML)'], 'FontSize', 14, 'FontWeight', 'bold');

if exist('elasticFD_record_out_path', 'var')
    exportgraphics(gcf, elasticFD_record_out_path, 'Resolution', 300);
end

if exist('elasticFD_data_out_path', 'var')
    save(elasticFD_data_out_path, ...
        'seis_vx', 'seis_vz', 'x_km', 'z_km', 't_axis', ...
        'vx_snaps', 'vz_snaps', 'plot_times', ...
        'vp', 'vs', 'rho', 'model_id', 'model_name', 'source_id', ...
        'nx_orig', 'nz_orig', 'dx', 'dz', 'dt', 'nt', 'f0', ...
        'xs_orig', 'zs_orig', 'receiver_depth', 'nabc');
end

%% ===================== 本地函数 =====================
function sigma = pml_sigma_1d(n, npml, d, vmax, Rcoef, m)
    sigma = zeros(1, n);
    sigma_max = - (m + 1) * vmax * log(Rcoef) / (2 * npml * d);

    for i = 1:n
        if i <= npml
            r = (npml - i + 1) / npml;
            sigma(i) = sigma_max * r^m;
        elseif i >= n - npml + 1
            r = (i - (n - npml)) / npml;
            sigma(i) = sigma_max * r^m;
        end
    end
end

function [a, b] = pml_ab(sigma, dt)
    b = exp(-sigma * dt);
    a = zeros(size(sigma));
    mask = sigma > 1e-12;
    a(mask) = (1 - b(mask)) ./ sigma(mask);
    a(~mask) = dt;
end

function apply_time_ticks(t_axis)
    yt = 0:0.1:max(t_axis);
    yticks(yt);

    yt_label = strings(size(yt));
    for ii = 1:length(yt)
        if abs(mod(yt(ii),0.5)) < 1e-8 || abs(mod(yt(ii),0.5)-0.5) < 1e-8
            yt_label(ii) = num2str(yt(ii), '%.1f');
        else
            yt_label(ii) = "";
        end
    end
    yticklabels(yt_label);
end
