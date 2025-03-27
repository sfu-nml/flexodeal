% The bash script "add_columns_to_qp_file.sh" can be used to assign
% constant properties throughout the muscle-tendon unit. However, for the
% case of a geometry with aponeurosis, we will have two sets of fibre
% orientations, so the quadrature point (QP) file must be updated
% accordingly.
%
% Author: Javier Almonacid
%         Neuromuscular Mechanics Laboratory
%         Simon Fraser University
%         March 27, 2025
clear; clc;

% Read quadrature point file, which can be generated using
%   ./flexodeal -QP_LIST_ONLY
qp_file = 'quadrature_point_data.csv';
df = readtable(qp_file);

% Assume that the fibres are oriented at a 15.3 degree angle with respect
% to the line of action, which we take as the x-axis.
initial_orientation_muscle = [cosd(15.3), 0, sind(15.3)];

% Furthermore, assume the fibres in the aponeurosis run in parallel. This 
% does not mean that the initial orientation vector is [1,0,0], since the
% aponeurosis is slanted with respect to the line of action (see geometry).
% After some geometry calculations, one can find that this angle is -4.7
% degrees.
initial_orientation_aponeurosis = [cosd(-4.7), 0, sind(-4.7)];

% Identify the rows in the table that correspond to muscle tissue
% (tissue_id = 1) and aponeurosis tissue (tissue_id = 2)
rows_mus = (df.tissue_id == 1);
rows_apo = (df.tissue_id == 2);

% Update rows with initial orientation information. First for muscle
% tissue:
df.muscle_fibre_orientation_x(rows_mus) = initial_orientation_muscle(1);
df.muscle_fibre_orientation_y(rows_mus) = initial_orientation_muscle(2);
df.muscle_fibre_orientation_z(rows_mus) = initial_orientation_muscle(3);
% then for aponeurosis tissue:
df.muscle_fibre_orientation_x(rows_apo) = initial_orientation_aponeurosis(1);
df.muscle_fibre_orientation_y(rows_apo) = initial_orientation_aponeurosis(2);
df.muscle_fibre_orientation_z(rows_apo) = initial_orientation_aponeurosis(3);

% If you ran the bash script "add_columns_to_qp_file.sh" to add the missing
% columns in your QP file, the next two statements below will, most likely, 
% not be executed. Otherwise, the columns below will be added:
if ~ismember('max_iso_stress_muscle', df.Properties.VariableNames)
    df.max_iso_stress_muscle = 200000 * ones(height(df), 1);
end
if ~ismember('fat_fraction', df.Properties.VariableNames)
    df.fat_fraction = 0.0 * ones(height(df), 1);
end

% Finally, overwrite the file with the updated data
writetable(df, qp_file);
