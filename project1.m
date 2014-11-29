% This file is the wrap-up script that calls all other training
% and testing scripts to output the result


% loading the raw data
load_data = load('project1_data.mat');
data = load_data.data;

% partitioning data into training, validation and testing
training_data = data(1:55698,:);
validation_data = data(55699:62661,:);
test_data = data(62662:69623,:);

% variables
M_cfs = 22; % model complexity for closed form
M_gd = 13; % model complexity for gradient descent
lambda_cfs = 0.01; % lambda for closed form
lambda_gd = 0.01; % lambda for gradient descent

% training the data using closed form solution
[rms_cfs_train,w_cfs] = train_cfs(training_data,M_cfs,lambda_cfs);

% testing the data using weight matrix
[rms_cfs] = test_cfs(test_data,M_cfs,lambda_cfs,w_cfs);

%rms_cfs = min([ermst,ermsv,erms]);


% training data for gradient descent solution
[rms_gd_train,w] = train_gd(training_data,M_gd,lambda_gd);

% testing the data using weight matrix
[rms_gd] = test_gd(test_data,M_gd,w);

%fprintf('%4.2f %4.2f\n',rms_cfs_train,rms_gd_train);

fprintf('My ubit name is %s\n','subhendu');
fprintf('My student number is %d \n',50097223);
fprintf('the model complexity M_cfs is %d\n', M_cfs);
fprintf('the model complexity M_gd is %d\n', M_gd);
fprintf('the regularization parameters lambda_cfs is %4.2f\n', lambda_cfs);
fprintf('the regularization parameters lambda_gd is %4.2f\n', lambda_gd);
fprintf('the root mean square error for the closed form solution is %4.2f\n',rms_cfs);
fprintf('the root mean square error for the gradient descent method is %4.2f\n',rms_gd);
