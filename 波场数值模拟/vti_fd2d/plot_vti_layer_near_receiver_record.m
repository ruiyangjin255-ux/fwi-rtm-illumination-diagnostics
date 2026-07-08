close all; clc;

dataPath = 'D:\ryjin\vti_fd2d\saved_data\vti_layer_near_receivers_data.mat';
outPath = 'D:\ryjin\paper_figures_source\vti\layer_three_stage\vti_layer_near_receivers_record_2s_vx_vz.png';

S = load(dataPath);

t = S.t_axis(:);
vx = S.seis_vx;
vz = S.seis_vz;
x = S.x_rec_km;

tgain = (t + 0.02).^1.0;
vx = vx .* tgain;
vz = vz .* tgain;

clipVx = localClip(vx, 99.5);
clipVz = localClip(vz, 99.5);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1500, 650]);

subplot(1,2,1);
imagesc(x, t, vx, [-clipVx clipVx]);
colormap(gca, gray);
set(gca, 'YDir', 'reverse', 'XAxisLocation', 'bottom', ...
    'FontSize', 15, 'LineWidth', 1.0);
xlabel('Distance (km)', 'FontSize', 17);
ylabel('Time (s)', 'FontSize', 17);
title('Shot Gather - Vx', 'FontSize', 19, 'FontWeight', 'normal');
axis tight;
xlim([min(x) max(x)]);
ylim([0 2.0]);
xticks(1.2:0.4:2.8);
yticks(0:0.5:2.0);

subplot(1,2,2);
imagesc(x, t, vz, [-clipVz clipVz]);
colormap(gca, gray);
set(gca, 'YDir', 'reverse', 'XAxisLocation', 'bottom', ...
    'FontSize', 15, 'LineWidth', 1.0);
xlabel('Distance (km)', 'FontSize', 17);
ylabel('Time (s)', 'FontSize', 17);
title('Shot Gather - Vz', 'FontSize', 19, 'FontWeight', 'normal');
axis tight;
xlim([min(x) max(x)]);
ylim([0 2.0]);
xticks(1.2:0.4:2.8);
yticks(0:0.5:2.0);

exportgraphics(fig, outPath, 'Resolution', 300);
close(fig);

fprintf('Near-source receiver record replotted: %s\n', outPath);

function c = localClip(a, pct)
    a = abs(a(:));
    a = a(isfinite(a));
    a = a(a > 0);
    if isempty(a)
        c = 1;
    else
        c = prctile(a, pct);
        if c <= 0
            c = max(a);
        end
    end
end
