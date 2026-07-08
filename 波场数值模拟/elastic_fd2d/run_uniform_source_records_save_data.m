close all; clc;

srcScript = 'D:\ryjin\elastic_fd2d\elasticFD_PML_source_fixed_data.m';
dataDir = 'D:\ryjin\elastic_fd2d\saved_data';
figDir = 'D:\ryjin\paper_figures_source\elastic';

if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

sourceIds = 1:4;
sourceTags = {'explosion', 'x_force', 'z_force', 'shear'};

for ii = 1:numel(sourceIds)
    elasticFD_batch_mode = true;
    elasticFD_model_id_override = 1;
    elasticFD_source_id_override = sourceIds(ii);
    elasticFD_nt_override = 1001;
    elasticFD_data_out_path = fullfile(dataDir, ...
        sprintf('uniform_%s_record_data.mat', sourceTags{ii}));
    elasticFD_record_out_path = fullfile(figDir, ...
        sprintf('uniform_%s_record_vx_vz.png', sourceTags{ii}));

    run(srcScript);

    clear elasticFD_batch_mode elasticFD_model_id_override ...
        elasticFD_source_id_override elasticFD_nt_override ...
        elasticFD_data_out_path elasticFD_record_out_path;
end

fprintf('Saved elastic record data under %s\n', dataDir);
