close all; clc;

outDir = 'D:\ryjin\paper_figures_source\vti\c13_series_wavefield_only';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

c11_gpa_base = 25.12;
c33_gpa_base = 17.99;
c44_gpa_base = 5.52;   % equals c55 in 2D P-SV implementation
c66_gpa_base = 5.85;
rho0_base = 2680;      % kg/m^3

% Target c13 values (GPa)
c13_list = [-17.5, -10.0, -5.8, 0.0, 1.5, 3.0, 5.8, 10.0, 17.5];

for i = 1:numel(c13_list)
    run_one_c13_wavefield_only( ...
        c13_list(i), ...
        c11_gpa_base, c33_gpa_base, c44_gpa_base, c66_gpa_base, rho0_base, ...
        outDir);
end

fprintf('Finished c13 wavefield-only series: %s\n', outDir);

function s = c13_tag(v)
if abs(v - round(v)) < 1e-8
    if v < 0
        s = sprintf('m%d', abs(round(v)));
    else
        s = sprintf('p%d', round(v));
    end
else
    if v < 0
        s = ['m' strrep(num2str(abs(v), '%.1f'), '.', 'p')];
    else
        s = ['p' strrep(num2str(v, '%.1f'), '.', 'p')];
    end
end
end

function run_one_c13_wavefield_only(c13_gpa, c11_gpa, c33_gpa, c44_gpa, c66_gpa, rho0, outDir)
% Convert fixed elastic parameters to Thomsen baseline pieces.
c11 = c11_gpa * 1e9;
c33 = c33_gpa * 1e9;
c44 = c44_gpa * 1e9;
c66 = c66_gpa * 1e9;
c13 = c13_gpa * 1e9;

vp0 = sqrt(c33 / rho0);
vs0 = sqrt(c44 / rho0);
epsi0 = (c11 / c33 - 1.0) / 2.0;
gamma0 = (c66 / c44 - 1.0) / 2.0;

% Invert delta from c13 relation used by the solver:
% c13 = sqrt(2*delta*c33*(c33-c44) + (c33-c44)^2) - c44
num = (c13 + c44)^2 - (c33 - c44)^2;
den = 2.0 * c33 * (c33 - c44);
delta0 = num / den;

tag = c13_tag(c13_gpa);
fprintf('Running c13 = %.1f GPa (delta=%.6f)\n', c13_gpa, delta0);

vtiFD_batch_mode = true;
vtiFD_model_id_override = 1;
vtiFD_source_id_override = 1;
vtiFD_nt_override = 1001;
vtiFD_f0_override = 10;
vtiFD_zs_orig_override = 201;
vtiFD_receiver_depth_orig_override = 181;
vtiFD_plot_times_override = 0.6;
vtiFD_fd_order_override = 6;
vtiFD_boundary_type_override = 'cpml';
vtiFD_no_total_title = true;
vtiFD_wave_title_mode = 'component_only';

vtiFD_thomsen_override = struct( ...
    'vp0', vp0, ...
    'vs0', vs0, ...
    'rho', rho0, ...
    'epsi', epsi0, ...
    'delta', delta0, ...
    'gamma', gamma0);

vtiFD_snapshot_out_path = fullfile(outDir, sprintf('vti_c13_%s_wavefield_vx_vz.png', tag));
vtiFD_record_out_path = '';
vtiFD_data_out_path = '';

run('D:\ryjin\vti_fd2d\vti_fd_chapter_data.m');
close all;
end
