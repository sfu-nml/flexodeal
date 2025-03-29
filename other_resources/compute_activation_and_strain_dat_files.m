% Compute strain profile from given function and compute activation from
% excitation profile using Zajac's transfer function.
% The data below was used in:
%
% Ross, S. A., Domínguez, S., Nigam, N., & Wakeling, J. M. (2021). 
% The energy of muscle contraction. III. Kinetic energy during cyclic 
% contractions. Frontiers in physiology, 12, 628819.
%
% Author: Javier Almonacid
%         Neuromuscular Mechanics Laboratory
%         Simon Fraser University
%         March 27, 2025
clear; close all; clc;

% Create time discretization
dt = 0.005;
Tend = 1.0;
time = 0:dt:Tend;

% Create strain profile
freq = 2;
strain_max = 0.05;
strain = @(t) strain_max*sin(2*pi*freq*t);
strain_vec = strain(time);

% Output strain profile
fid = fopen('control_points_strain.dat','w');
for i = 1:length(time)
    fprintf(fid, '%.4f\t%.6f\n', time(i), strain_vec(i));
end
fclose(fid);

% Create excitation profile
emax = 1;
duty_cycle = 0.3;
excitation = emax * square_wave(time-0.81*Tend/8, freq, duty_cycle);
plot(time, excitation)

% Define parameters for Zajac's ODE and compute activation
Tact = 0.045;
beta = 0.6;
activation = zajac_activation_from_emg(time, excitation, Tact, beta);
% Zero-out negative entries
idx = find(activation < 0);
activation(idx) = 0;

% Output activation profile
fid = fopen('control_points_activation.dat','w');
for i = 1:length(time)
    fprintf(fid, '%.4f\t%.6f\n', time(i), activation(i));
end
fclose(fid);

% Plot results
subplot(2,1,1)
plot(time, strain_vec, 'LineWidth', 2)
subplot(2,1,2)
plot(time, excitation, 'LineWidth', 2)
hold on
plot(time, activation, 'LineWidth', 2)

% The function below has been retrieved from:
% https://github.com/javieralmonacid/multibody-muscle-1d/blob/main/tools/zajac_activation_from_emg.m
function recorded_activation = zajac_activation_from_emg(recorded_time, ...
                                                         recorded_emg, ...
                                                         Tact, ...
                                                         beta)
% ZAJAC_ACTIVATION_FROM_EMG Computes activation function using EMG data
%
% Parameters
% --------------
%   recorded_time: a vector of time steps (in seconds).
%   recorded_emg: a vector of the same length as recorded_time with
%   excitation datapoints.
%   Tact: tau_act in Zajac's ODE.
%   beta: beta in Zajac's ODE.
%
% Output
% ---------------
%   recorded_activation: computed activation at time steps given by
%   recorded_time using Zajac's transfer function.
%
% Copyright (C) 2024
% Javier Almonacid
% Neuromuscular Mechanics Laboratory
% Simon Fraser University

    excitation = @(t) interp1(recorded_time, recorded_emg, t);
    zajac_ode = @(t,a) -a*(beta+excitation(t)*(1-beta))/Tact + excitation(t)/Tact;
    sol_zajac = ode23s(zajac_ode, [recorded_time(1), recorded_time(end)], 0.0);
    activation_function = @(t) deval(sol_zajac, t);
    recorded_activation = activation_function(recorded_time);
end

function s = square_wave(t,freq,duty_cycle)
    % Create a square wave without the need of Matlab's signal processing
    % toolbox
    tmp = mod(t,1/freq);
    w0 = (1/freq)*duty_cycle; % Pulse width
    s = (tmp < w0);
    s = 1.0*s; % So that the output is not logical
end