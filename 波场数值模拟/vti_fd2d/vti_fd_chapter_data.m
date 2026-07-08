if ~exist('vtiFD_batch_mode', 'var') || ~vtiFD_batch_mode
    clear; clc; close all;
else
    clc; close all;
end

%% ===================== 1. 鍙傛暟 =====================
if ~exist('vtiFD_model_id_override', 'var')
    vtiFD_model_id_override = 2;
end
model_id  = vtiFD_model_id_override;
if ~exist('vtiFD_source_id_override', 'var')
    vtiFD_source_id_override = 1;
end
source_id = vtiFD_source_id_override;

% ===== 缃戞牸鍙傛暟 =====
nx_orig = 401;
nz_orig = 401;
dx = 10;                     % m
dz = 10;                     % m

% ===== 鏃堕棿鍙傛暟 =====
nt = 2000;
if exist('vtiFD_nt_override', 'var')
    nt = vtiFD_nt_override;
end
dt = 0.001;                % s
f0 = 10;                    % Hz
if exist('vtiFD_f0_override', 'var')
    f0 = vtiFD_f0_override;
end
t0 = 1 / f0;

% ===== CPML 鍙傛暟 =====
nabc       = 40;
Rcoef      = 1e-6;
cpml_order = 3;
kappa_max  = 8.0;
alpha_max  = pi * f0;
boundary_type = 'cpml';
if exist('vtiFD_boundary_type_override', 'var')
    boundary_type = lower(vtiFD_boundary_type_override);
end
if strcmp(boundary_type, 'pml')
    kappa_max = 1.0;
    alpha_max = 0.0;
elseif ~strcmp(boundary_type, 'cpml')
    error('Unsupported boundary type: %s. Use pml or cpml.', boundary_type);
end

nx = nx_orig + 2*nabc;
nz = nz_orig + 2*nabc;

% ===== 鐐偣 / 妫€娉㈠櫒 =====
zs_orig = 201;
if exist('vtiFD_zs_orig_override', 'var')
    zs_orig = vtiFD_zs_orig_override;
end
xs_orig = round((nx_orig+1)/2);
receiver_depth_orig = 5;
if exist('vtiFD_receiver_depth_orig_override', 'var')
    receiver_depth_orig = vtiFD_receiver_depth_orig_override;
end

zs = zs_orig + nabc;
xs = xs_orig + nabc;
receiver_depth = receiver_depth_orig + nabc;
receiver_x_orig = 1:nx_orig;
if exist('vtiFD_receiver_x_orig_override', 'var')
    receiver_x_orig = vtiFD_receiver_x_orig_override;
end
receiver_x_orig = receiver_x_orig(receiver_x_orig >= 1 & receiver_x_orig <= nx_orig);
if isempty(receiver_x_orig)
    error('receiver_x_orig is empty after bounds check.');
end

% ===== 蹇収鏃跺埢 =====
plot_times = [0.6];
if exist('vtiFD_plot_times_override', 'var')
    plot_times = vtiFD_plot_times_override;
end
plot_steps = round(plot_times / dt);

% ===== 鏄剧ず鍙傛暟 =====
color_mode = 2;       % 1: Vx/Vz鍏辫壊鏍? 2: 鍚勮嚜鐙珛鑹叉爣
clip_ratio = 0.995;

%% ===================== 2. VTI 妯″瀷鍙傛暟锛堢洿鎺ヨ緭鍏?Cij锛?=====================
% 鍗曚綅缁熶竴涓?Pa
rho = zeros(nz, nx);
c11 = zeros(nz, nx);
c13 = zeros(nz, nx);
c33 = zeros(nz, nx);
c44 = zeros(nz, nx);   % 2D x-z VTI 涓瓑鏁堜娇鐢?c44鈮坈55
c66 = zeros(nz, nx);   % 褰撳墠 2D 涓ゅ垎閲忎富鏂圭▼閲屽熀鏈笉鐢紝浣嗕繚鐣?

[X, Z] = meshgrid(1:nx, 1:nz);

switch model_id
    case 1
        model_name = 'Homogeneous VTI (Thomsen)';

        % Thomsen-style parameterization for the uniform VTI reference model.
        vp0 = 2590.8852863768743;
        vs0 = 1435.1660156711187;
        rho0 = 2680;
        epsi0 = 0.19816564758198998;
        delta0 = -0.23674495275598628;
        gamma0 = 0.029891304347826088;

        if exist('vtiFD_thomsen_override', 'var')
            th = vtiFD_thomsen_override;
            if isfield(th, 'vp0'), vp0 = th.vp0; end
            if isfield(th, 'vs0'), vs0 = th.vs0; end
            if isfield(th, 'rho'), rho0 = th.rho; end
            if isfield(th, 'epsi'), epsi0 = th.epsi; end
            if isfield(th, 'delta'), delta0 = th.delta; end
            if isfield(th, 'gamma'), gamma0 = th.gamma; end
        end

        c33(:) = rho0 * vp0^2;
        c44(:) = rho0 * vs0^2;
        c11(:) = c33(:) * (1 + 2 * epsi0);
        tmp = 2 * delta0 .* c33(:) .* (c33(:) - c44(:)) + (c33(:) - c44(:)).^2;
        c13(:) = sqrt(max(tmp, 0)) - c44(:);
        c66(:) = c44(:) * (1 + 2 * gamma0);
        rho(:) = rho0;

    case 2
        model_name = 'Two-layer VTI';

        idx = Z > (100 + nabc);

        % ===== 涓婂眰锛氭寜璁烘枃VTI灞傚弬鏁伴鏍?=====
        c11(~idx) = 25.12e9;
        c13(~idx) = 3.0e9;
        c33(~idx) = 18.2e9;
        c44(~idx) = 5.52e9;
        c66(~idx) = 5.85e9;
        rho(~idx) = 2200;

        % ===== 涓嬪眰锛氱粰涓€缁勬洿寮轰竴鐐圭殑VTI鍙傛暟锛屼究浜庣湅鐣岄潰鏁堝簲 =====
        c11(idx) = 32.0e9;
        c13(idx) = 6.0e9;
        c33(idx) = 24.0e9;
        c44(idx) = 7.5e9;
        c66(idx) = 8.0e9;
        rho(idx) = 2450;

    case 5
        model_name = 'Two-layer VTI (Thomsen)';

        idx = Z > (100 + nabc);

        vp0 = zeros(nz, nx);
        vs0 = zeros(nz, nx);
        epsi0 = zeros(nz, nx);
        delta0 = zeros(nz, nx);

        % Upper layer: velocity-density-Thomsen parameter input.
        vp0(~idx) = 2400;
        vs0(~idx) = 1200;
        rho(~idx) = 2200;
        epsi0(~idx) = 0.20;
        delta0(~idx) = 0.10;

        % Lower layer: velocity-density-Thomsen parameter input.
        vp0(idx) = 3200;
        vs0(idx) = 1800;
        rho(idx) = 2500;
        epsi0(idx) = 0.30;
        delta0(idx) = 0.15;

        c33(:) = rho(:) .* vp0(:).^2;
        c44(:) = rho(:) .* vs0(:).^2;
        c11(:) = c33(:) .* (1 + 2 * epsi0(:));
        tmp = 2 .* delta0(:) .* c33(:) .* (c33(:) - c44(:)) + ...
              (c33(:) - c44(:)).^2;
        c13(:) = sqrt(max(tmp, 0)) - c44(:);
        c66(:) = c44(:);

    case 3
        model_name = 'Homogeneous isotropic elastic';

        vp = 2500;
        vs = 1200;
        rho(:) = 2200;
        c33(:) = rho .* vp.^2;
        c11(:) = c33;
        c44(:) = rho .* vs.^2;
        c66(:) = c44;
        c13(:) = rho .* (vp.^2 - 2*vs.^2);

    case 4
        model_name = 'Complex salt VTI';

        depth_km = (Z - nabc - 1) * dz / 1000;
        x_km_grid = (X - nabc - 1) * dx / 1000;

        vp_bg = 2200 + 350 * depth_km;
        vs_bg = 0.55 * vp_bg;
        rho_bg = 2100 + 80 * depth_km;
        epsi_bg = 0.10 + 0.03 * depth_km;
        delta_bg = 0.05 + 0.02 * depth_km;

        salt_top = 0.85 + 0.20 * cos((x_km_grid - 2.0) * pi);
        salt_bottom = 2.20 - 0.15 * cos((x_km_grid - 2.0) * pi);
        salt_mask = abs(x_km_grid - 2.0) < 0.75 & depth_km > salt_top & depth_km < salt_bottom;

        vp0 = vp_bg;
        vs0 = vs_bg;
        rho_tmp = rho_bg;
        epsi_tmp = epsi_bg;
        delta_tmp = delta_bg;

        vp0(salt_mask) = 4500;
        vs0(salt_mask) = 2600;
        rho_tmp(salt_mask) = 2400;
        epsi_tmp(salt_mask) = 0.02;
        delta_tmp(salt_mask) = 0.01;

        rho(:) = rho_tmp;
        c33(:) = rho(:) .* vp0(:).^2;
        c44(:) = rho(:) .* vs0(:).^2;
        c11(:) = c33(:) .* (1 + 2*epsi_tmp(:));
        tmp = 2 .* delta_tmp(:) .* c33(:) .* (c33(:) - c44(:)) + (c33(:) - c44(:)).^2;
        c13(:) = sqrt(max(tmp, 0)) - c44(:);
        c66(:) = c44(:);
end

%% ===================== 3. 鍙傛暟绋冲畾鎬ф鏌?=====================
% 鐗╃悊绋冲畾鎬?
if any(c11(:) <= 0) || any(c33(:) <= 0) || any(c44(:) <= 0)
    error('Invalid stiffness parameters: c11/c33/c44 must be positive.');
end

if any((c11(:).*c33(:) - c13(:).^2) <= 0)
    error('Invalid VTI parameters: c11*c33 - c13^2 <= 0.');
end

% 鏁板€肩ǔ瀹氭€?
vmax = max(sqrt(c11(:) ./ rho(:)));   % 鍙?x 鏂瑰悜 qP 杩戜技涓婇檺
cfl  = vmax * dt / min(dx, dz);
fprintf('CFL = %.4f\n', cfl);

if cfl >= 0.50
    error('CFL is too large: %.4f. Reduce dt or increase dx/dz.', cfl);
end

%% ===================== 4. 浜ら敊缃戞牸鍙橀噺 =====================
% sxx,szz 鍦ㄤ腑蹇冪偣 (nz x nx)
% vx      鍦?x 鍗婃牸鐐?(nz x (nx+1))
% vz      鍦?z 鍗婃牸鐐?((nz+1) x nx)
% sxz     鍦ㄨ鐐瑰崐鏍?((nz+1) x (nx+1))

sxx = zeros(nz,   nx);
szz = zeros(nz,   nx);
vx  = zeros(nz,   nx+1);
vz  = zeros(nz+1, nx);
sxz = zeros(nz+1, nx+1);

% 鍗曠偖璁板綍
seis_vx = zeros(nt, numel(receiver_x_orig));
seis_vz = zeros(nt, numel(receiver_x_orig));

% 蹇収
vx_snaps = cell(1, numel(plot_steps));
vz_snaps = cell(1, numel(plot_steps));
sid = 1;

%% ===================== 5. 鍏樁宸垎绯绘暟 =====================
fd_order = 6;
if exist('vtiFD_fd_order_override', 'var')
    fd_order = vtiFD_fd_order_override;
end
switch fd_order
    case 4
        c1 = 9/8;
        c2 = -1/24;
        c3 = 0;
    case 6
        c1 = 75/64;
        c2 = -25/384;
        c3 = 3/640;
    otherwise
        error('Unsupported finite-difference order: %d. Use 4 or 6.', fd_order);
end

%% ===================== 6. 鏋勯€?CPML 1D 鍓栭潰 =====================
[sigma_x_c, kappa_x_c, alpha_x_c, sigma_x_h, kappa_x_h, alpha_x_h] = ...
    make_cpml_profile_1d(nx, nabc, dx, vmax, f0, Rcoef, cpml_order, kappa_max, alpha_max);

[sigma_z_c, kappa_z_c, alpha_z_c, sigma_z_h, kappa_z_h, alpha_z_h] = ...
    make_cpml_profile_1d(nz, nabc, dz, vmax, f0, Rcoef, cpml_order, kappa_max, alpha_max);

sigma_z_c = sigma_z_c(:);  kappa_z_c = kappa_z_c(:);  alpha_z_c = alpha_z_c(:);
sigma_z_h = sigma_z_h(:);  kappa_z_h = kappa_z_h(:);  alpha_z_h = alpha_z_h(:);

%% ===================== 7. 涓嶅悓鍙橀噺浣嶇疆涓婄殑 CPML 鍙傛暟 =====================
% vx: (nz, nx+1)   x-half, z-center
sigx_vx = repmat(sigma_x_h, nz, 1);
kapx_vx = repmat(kappa_x_h, nz, 1);
alpx_vx = repmat(alpha_x_h, nz, 1);

sigz_vx = repmat(sigma_z_c, 1, nx+1);
kapz_vx = repmat(kappa_z_c, 1, nx+1);
alpz_vx = repmat(alpha_z_c, 1, nx+1);

% vz: (nz+1, nx)  x-center, z-half
sigx_vz = repmat(sigma_x_c, nz+1, 1);
kapx_vz = repmat(kappa_x_c, nz+1, 1);
alpx_vz = repmat(alpha_x_c, nz+1, 1);

sigz_vz = repmat(sigma_z_h, 1, nx);
kapz_vz = repmat(kappa_z_h, 1, nx);
alpz_vz = repmat(alpha_z_h, 1, nx);

% center stresses: (nz, nx)
sigx_c = repmat(sigma_x_c, nz, 1);
kapx_c = repmat(kappa_x_c, nz, 1);
alpx_c = repmat(alpha_x_c, nz, 1);

sigz_c = repmat(sigma_z_c, 1, nx);
kapz_c = repmat(kappa_z_c, 1, nx);
alpz_c = repmat(alpha_z_c, 1, nx);

% sxz: (nz+1, nx+1)  x-half, z-half
sigx_sxz = repmat(sigma_x_h, nz+1, 1);
kapx_sxz = repmat(kappa_x_h, nz+1, 1);
alpx_sxz = repmat(alpha_x_h, nz+1, 1);

sigz_sxz = repmat(sigma_z_h, 1, nx+1);
kapz_sxz = repmat(kappa_z_h, 1, nx+1);
alpz_sxz = repmat(alpha_z_h, 1, nx+1);

%% ===================== 8. CPML a/b 绯绘暟 =====================
[a_vx_x, b_vx_x] = make_cpml_ab(sigx_vx,   kapx_vx,   alpx_vx,   dt);
[a_vx_z, b_vx_z] = make_cpml_ab(sigz_vx,   kapz_vx,   alpz_vx,   dt);

[a_vz_x, b_vz_x] = make_cpml_ab(sigx_vz,   kapx_vz,   alpx_vz,   dt);
[a_vz_z, b_vz_z] = make_cpml_ab(sigz_vz,   kapz_vz,   alpz_vz,   dt);

[a_c_x,  b_c_x ] = make_cpml_ab(sigx_c,    kapx_c,    alpx_c,    dt);
[a_c_z,  b_c_z ] = make_cpml_ab(sigz_c,    kapz_c,    alpz_c,    dt);

[a_sxz_x, b_sxz_x] = make_cpml_ab(sigx_sxz, kapx_sxz, alpx_sxz, dt);
[a_sxz_z, b_sxz_z] = make_cpml_ab(sigz_sxz, kapz_sxz, alpz_sxz, dt);

%% ===================== 9. CPML 璁板繂鍙橀噺 =====================
psi_vx_x  = zeros(nz,   nx+1);
psi_vx_z  = zeros(nz,   nx+1);

psi_vz_x  = zeros(nz+1, nx);
psi_vz_z  = zeros(nz+1, nx);

psi_c_x   = zeros(nz, nx);
psi_c_z   = zeros(nz, nx);

psi_sxz_x = zeros(nz+1, nx+1);
psi_sxz_z = zeros(nz+1, nx+1);

%% ===================== 10. 闇囨簮 =====================
t_vec = (0:nt-1) * dt;
arg = (pi * f0 * (t_vec - t0)).^2;
wavelet = (1 - 2 * arg) .* exp(-arg);

%% ===================== 11. 鏃堕棿鎺ㄨ繘 =====================
for it = 1:nt

    % ---------- 11.1 鏇存柊 vx ----------
    jv = 4:nz-3;
    iv = 4:nx-3;

    raw_dsxx_dx = ( ...
        c1 * (sxx(jv, iv)   - sxx(jv, iv-1)) + ...
        c2 * (sxx(jv, iv+1) - sxx(jv, iv-2)) + ...
        c3 * (sxx(jv, iv+2) - sxx(jv, iv-3)) ) / dx;

    raw_dsxz_dz = ( ...
        c1 * (sxz(jv+1, iv) - sxz(jv,   iv)) + ...
        c2 * (sxz(jv+2, iv) - sxz(jv-1, iv)) + ...
        c3 * (sxz(jv+3, iv) - sxz(jv-2, iv)) ) / dz;

    psi_vx_x(jv, iv) = b_vx_x(jv, iv) .* psi_vx_x(jv, iv) + a_vx_x(jv, iv) .* raw_dsxx_dx;
    psi_vx_z(jv, iv) = b_vx_z(jv, iv) .* psi_vx_z(jv, iv) + a_vx_z(jv, iv) .* raw_dsxz_dz;

    dsxx_dx = raw_dsxx_dx ./ kapx_vx(jv, iv) + psi_vx_x(jv, iv);
    dsxz_dz = raw_dsxz_dz ./ kapz_vx(jv, iv) + psi_vx_z(jv, iv);

    rho_vx = 0.5 * (rho(jv, iv) + rho(jv, iv-1));
    vx(jv, iv) = vx(jv, iv) + (dt ./ rho_vx) .* (dsxx_dx + dsxz_dz);

    % ---------- 11.2 鏇存柊 vz ----------
    jv2 = 4:nz-3;
    iv2 = 4:nx-3;

    raw_dsxz_dx = ( ...
        c1 * (sxz(jv2, iv2+1) - sxz(jv2, iv2))   + ...
        c2 * (sxz(jv2, iv2+2) - sxz(jv2, iv2-1)) + ...
        c3 * (sxz(jv2, iv2+3) - sxz(jv2, iv2-2)) ) / dx;

    raw_dszz_dz = ( ...
        c1 * (szz(jv2,   iv2) - szz(jv2-1, iv2)) + ...
        c2 * (szz(jv2+1, iv2) - szz(jv2-2, iv2)) + ...
        c3 * (szz(jv2+2, iv2) - szz(jv2-3, iv2)) ) / dz;

    psi_vz_x(jv2, iv2) = b_vz_x(jv2, iv2) .* psi_vz_x(jv2, iv2) + a_vz_x(jv2, iv2) .* raw_dsxz_dx;
    psi_vz_z(jv2, iv2) = b_vz_z(jv2, iv2) .* psi_vz_z(jv2, iv2) + a_vz_z(jv2, iv2) .* raw_dszz_dz;

    dsxz_dx = raw_dsxz_dx ./ kapx_vz(jv2, iv2) + psi_vz_x(jv2, iv2);
    dszz_dz = raw_dszz_dz ./ kapz_vz(jv2, iv2) + psi_vz_z(jv2, iv2);

    rho_vz = 0.5 * (rho(jv2, iv2) + rho(jv2-1, iv2));
    vz(jv2, iv2) = vz(jv2, iv2) + (dt ./ rho_vz) .* (dsxz_dx + dszz_dz);

    % ---------- 11.3 鍔犻渿婧?----------
    if it <= round(2*t0/dt)
        s = wavelet(it);

        switch source_id
            case 1  % 鐖嗙偢婧?
                sxx(zs, xs) = sxx(zs, xs) + s;
                szz(zs, xs) = szz(zs, xs) + s;

            case 2  % 姘村钩鍔涙簮
                vx(zs, xs)   = vx(zs, xs)   + 0.5 * s;
                vx(zs, xs+1) = vx(zs, xs+1) + 0.5 * s;

            case 3  % 鍨傜洿鍔涙簮
                vz(zs,   xs) = vz(zs,   xs) + 0.5 * s;
                vz(zs+1, xs) = vz(zs+1, xs) + 0.5 * s;

            case 4  % 鍓垏婧?
                sxz(zs,   xs)   = sxz(zs,   xs)   + 0.25 * s;
                sxz(zs+1, xs)   = sxz(zs+1, xs)   + 0.25 * s;
                sxz(zs,   xs+1) = sxz(zs,   xs+1) + 0.25 * s;
                sxz(zs+1, xs+1) = sxz(zs+1, xs+1) + 0.25 * s;
        end
    end

    % ---------- 11.4 鏇存柊 sxx / szz ----------
    js = 4:nz-3;
    is = 4:nx-3;

    raw_dvx_dx = ( ...
        c1 * (vx(js, is+1) - vx(js, is))   + ...
        c2 * (vx(js, is+2) - vx(js, is-1)) + ...
        c3 * (vx(js, is+3) - vx(js, is-2)) ) / dx;

    raw_dvz_dz = ( ...
        c1 * (vz(js+1, is) - vz(js,   is)) + ...
        c2 * (vz(js+2, is) - vz(js-1, is)) + ...
        c3 * (vz(js+3, is) - vz(js-2, is)) ) / dz;

    psi_c_x(js, is) = b_c_x(js, is) .* psi_c_x(js, is) + a_c_x(js, is) .* raw_dvx_dx;
    psi_c_z(js, is) = b_c_z(js, is) .* psi_c_z(js, is) + a_c_z(js, is) .* raw_dvz_dz;

    dvx_dx = raw_dvx_dx ./ kapx_c(js, is) + psi_c_x(js, is);
    dvz_dz = raw_dvz_dz ./ kapz_c(js, is) + psi_c_z(js, is);

    sxx(js, is) = sxx(js, is) + dt * (c11(js,is) .* dvx_dx + c13(js,is) .* dvz_dz);
    szz(js, is) = szz(js, is) + dt * (c13(js,is) .* dvx_dx + c33(js,is) .* dvz_dz);

    % ---------- 11.5 鏇存柊 sxz ----------
    jt  = 4:nz-3;
    itx = 4:nx-3;

    raw_dvx_dz = ( ...
        c1 * (vx(jt,   itx) - vx(jt-1, itx)) + ...
        c2 * (vx(jt+1, itx) - vx(jt-2, itx)) + ...
        c3 * (vx(jt+2, itx) - vx(jt-3, itx)) ) / dz;

    raw_dvz_dx = ( ...
        c1 * (vz(jt, itx)   - vz(jt, itx-1)) + ...
        c2 * (vz(jt, itx+1) - vz(jt, itx-2)) + ...
        c3 * (vz(jt, itx+2) - vz(jt, itx-3)) ) / dx;

    psi_sxz_z(jt, itx) = b_sxz_z(jt, itx) .* psi_sxz_z(jt, itx) + a_sxz_z(jt, itx) .* raw_dvx_dz;
    psi_sxz_x(jt, itx) = b_sxz_x(jt, itx) .* psi_sxz_x(jt, itx) + a_sxz_x(jt, itx) .* raw_dvz_dx;

    dvx_dz = raw_dvx_dz ./ kapz_sxz(jt, itx) + psi_sxz_z(jt, itx);
    dvz_dx = raw_dvz_dx ./ kapx_sxz(jt, itx) + psi_sxz_x(jt, itx);

    c44_sxz = 0.25 * (c44(jt,itx) + c44(jt-1,itx) + c44(jt,itx-1) + c44(jt-1,itx-1));
    sxz(jt, itx) = sxz(jt, itx) + dt * c44_sxz .* (dvx_dz + dvz_dx);

    % ---------- 11.6 璁板綍 ----------
    vx_c = 0.5 * (vx(:,1:nx) + vx(:,2:nx+1));
    vz_c = 0.5 * (vz(1:nz,:) + vz(2:nz+1,:));

    seis_vx(it,:) = vx_c(receiver_depth, nabc + receiver_x_orig);
    seis_vz(it,:) = vz_c(receiver_depth, nabc + receiver_x_orig);

    % ---------- 11.7 淇濆瓨蹇収 ----------
    if sid <= numel(plot_steps) && it == plot_steps(sid)
        vx_snaps{sid} = vx_c(nabc+1:end-nabc, nabc+1:end-nabc);
        vz_snaps{sid} = vz_c(nabc+1:end-nabc, nabc+1:end-nabc);
        sid = sid + 1;
    end

    % ---------- 11.8 杩涘害鏄剧ず ----------
    if mod(it, 100) == 0
        fprintf('it = %d / %d\n', it, nt);
    end
end

%% ===================== 12. 鍧愭爣 =====================
x_km = (0:nx_orig-1) * dx / 1000;
z_km = (0:nz_orig-1) * dz / 1000;
x_rec_km = (receiver_x_orig - 1) * dx / 1000;
t_axis = (0:nt-1) * dt;

m = 256;
seismic_cmap = [linspace(0,1,m/2)', linspace(0,1,m/2)', ones(m/2,1); ...
                ones(m/2,1), linspace(1,0,m/2)', linspace(1,0,m/2)'];

%% ===================== 13. 娉㈠満蹇収 =====================
nshot = numel(plot_times);
fig_wave = figure('Visible','off','Color','w', 'Position', [100, 100, 1200, 420*nshot]);
tl = tiledlayout(nshot, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for k = 1:nshot
    if color_mode == 1
        clip_val = robust_abs_clip([vx_snaps{k}(:); vz_snaps{k}(:)], clip_ratio);
        clip_vx = clip_val;
        clip_vz = clip_val;
    else
        clip_vx = robust_abs_clip(vx_snaps{k}, clip_ratio);
        clip_vz = robust_abs_clip(vz_snaps{k}, clip_ratio);
    end

    nexttile;
    imagesc(x_km, z_km, vx_snaps{k});
    set(gca, 'YDir', 'reverse');
    axis image;
    colormap(gca, seismic_cmap);
    caxis([-clip_vx, clip_vx]);
    if ~exist('vtiFD_wave_title_mode', 'var')
        vtiFD_wave_title_mode = 'component_only';
    end
    if strcmpi(vtiFD_wave_title_mode, 'component_only')
        title('Vx');
    else
        title(['Vx at t=', num2str(plot_times(k),'%.2f'), 's']);
    end
    xlabel('Distance (km)');
    ylabel('Depth (km)');
    colorbar;

    nexttile;
    imagesc(x_km, z_km, vz_snaps{k});
    set(gca, 'YDir', 'reverse');
    axis image;
    colormap(gca, seismic_cmap);
    caxis([-clip_vz, clip_vz]);
    if strcmpi(vtiFD_wave_title_mode, 'component_only')
        title('Vz');
    else
        title(['Vz at t=', num2str(plot_times(k),'%.2f'), 's']);
    end
    xlabel('Distance (km)');
    ylabel('Depth (km)');
    colorbar;
end

if ~exist('vtiFD_no_total_title', 'var') || ~vtiFD_no_total_title
    title(tl, ['Wavefield Snapshots - ', model_name, ' (6th-order FD + CPML)'], ...
        'FontSize', 14, 'FontWeight', 'bold');
end

%% ===================== 14. 鍦伴渿鍗曠偖璁板綍 =====================
seis_vx_disp = seis_vx;
seis_vz_disp = seis_vz;

% 杞诲井鏃堕棿澧炵泭锛堝彧鐢ㄤ簬鏄剧ず锛?
tgain = (t_axis(:) + 0.02).^1.0;
seis_vx_disp = seis_vx_disp .* tgain;
seis_vz_disp = seis_vz_disp .* tgain;

% 鍏ㄥ眬褰掍竴鍖?
ampx = max(abs(seis_vx_disp(:)));
ampz = max(abs(seis_vz_disp(:)));
if ampx == 0, ampx = 1; end
if ampz == 0, ampz = 1; end

seis_vx_disp = seis_vx_disp / ampx;
seis_vz_disp = seis_vz_disp / ampz;

fig_record = figure('Visible','off','Color','w', 'Position', [140, 100, 1400, 650]);

subplot(1,2,1);
imagesc(x_rec_km, t_axis, seis_vx_disp);
colormap(gca, gray);
set(gca,'YDir','reverse');
set(gca,'XAxisLocation','bottom');
caxis([-0.45, 0.45]);
xlabel('Distance (km)');
ylabel('Time (s)');
title('Shot Gather - Vx');
apply_record_axis_format(x_rec_km, t_axis);

subplot(1,2,2);
imagesc(x_rec_km, t_axis, seis_vz_disp);
colormap(gca, gray);
set(gca,'YDir','reverse');
set(gca,'XAxisLocation','bottom');
caxis([-0.45, 0.45]);
xlabel('Distance (km)');
ylabel('Time (s)');
title('Shot Gather - Vz');
apply_record_axis_format(x_rec_km, t_axis);

if ~exist('vtiFD_no_total_title', 'var') || ~vtiFD_no_total_title
    sgtitle(['Seismic Shot Gathers - ', model_name, ' (', num2str(fd_order), 'th-order FD + ', upper(boundary_type), ')'], ...
        'FontSize', 14, 'FontWeight', 'bold');
end

if exist('vtiFD_snapshot_out_path', 'var') && ~isempty(vtiFD_snapshot_out_path)
    exportgraphics(fig_wave, vtiFD_snapshot_out_path, 'Resolution', 300);
end
if exist('vtiFD_record_out_path', 'var') && ~isempty(vtiFD_record_out_path)
    exportgraphics(fig_record, vtiFD_record_out_path, 'Resolution', 300);
end
if exist('vtiFD_data_out_path', 'var') && ~isempty(vtiFD_data_out_path)
    save(vtiFD_data_out_path, 'seis_vx', 'seis_vz', 'vx_snaps', 'vz_snaps', ...
        'x_km', 'z_km', 'x_rec_km', 'receiver_x_orig', 't_axis', 'plot_times', 'model_id', 'model_name', ...
        'source_id', 'rho', 'c11', 'c13', 'c33', 'c44', 'c66', 'dx', 'dz', ...
        'dt', 'nt', 'f0', 'xs_orig', 'zs_orig', 'receiver_depth_orig', 'fd_order', 'boundary_type', '-v7.3');
end

%% ===================== 鏈湴鍑芥暟 =====================
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

function apply_record_axis_format(x_rec_km, t_axis)
    xmin = min(x_rec_km);
    xmax = max(x_rec_km);
    xlim([xmin xmax]);
    xt = ceil(xmin):1:floor(xmax);
    if isempty(xt)
        xt = linspace(xmin, xmax, 5);
    end
    xticks(xt);
    xticklabels(compose('%.0f', xt));

    tmax = max(t_axis);
    yt = 0:0.5:tmax;
    yticks(yt);
    yticklabels(compose('%.1f', yt));
end

function clipv = robust_abs_clip(A, ratio)
    v = sort(abs(A(:)));
    if isempty(v)
        clipv = 1;
        return;
    end
    idx = max(1, min(numel(v), round(ratio * numel(v))));
    clipv = v(idx);
    if clipv <= 0
        clipv = max(v);
    end
    if clipv <= 0
        clipv = 1;
    end
end

function [sigma_c, kappa_c, alpha_c, sigma_h, kappa_h, alpha_h] = ...
    make_cpml_profile_1d(n, npml, d, vmax, f0, Rcoef, m, kappa_max, alpha_max)

    sigma_c = zeros(1, n);
    kappa_c = ones(1, n);
    alpha_c = zeros(1, n);

    if npml <= 0
        sigma_h = zeros(1, n+1);
        kappa_h = ones(1, n+1);
        alpha_h = zeros(1, n+1);
        return;
    end

    sigma_max = -(m + 1) * vmax * log(Rcoef) / (2 * npml * d);

    for i = 1:n
        if i <= npml
            x = (npml - i + 1) / npml;
            sigma_c(i) = sigma_max * x^m;
            kappa_c(i) = 1 + (kappa_max - 1) * x^m;
            alpha_c(i) = alpha_max * (1 - x);
        elseif i >= n - npml + 1
            x = (i - (n - npml)) / npml;
            sigma_c(i) = sigma_max * x^m;
            kappa_c(i) = 1 + (kappa_max - 1) * x^m;
            alpha_c(i) = alpha_max * (1 - x);
        end
    end

    sigma_h = zeros(1, n+1);
    kappa_h = ones(1, n+1);
    alpha_h = zeros(1, n+1);

    sigma_h(2:n) = 0.5 * (sigma_c(1:n-1) + sigma_c(2:n));
    kappa_h(2:n) = 0.5 * (kappa_c(1:n-1) + kappa_c(2:n));
    alpha_h(2:n) = 0.5 * (alpha_c(1:n-1) + alpha_c(2:n));

    sigma_h(1)   = sigma_c(1);
    sigma_h(n+1) = sigma_c(n);

    kappa_h(1)   = kappa_c(1);
    kappa_h(n+1) = kappa_c(n);

    alpha_h(1)   = alpha_c(1);
    alpha_h(n+1) = alpha_c(n);
end

function [a, b] = make_cpml_ab(sigma, kappa, alpha, dt)
    b = exp(-(sigma ./ kappa + alpha) * dt);
    a = zeros(size(sigma));

    mask = sigma > 1e-12;
    a(mask) = sigma(mask) .* (b(mask) - 1) ./ ...
              (kappa(mask) .* (sigma(mask) + kappa(mask) .* alpha(mask)));
end

