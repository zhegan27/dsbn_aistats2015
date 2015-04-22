clear all; close all; clc;
randn('state',100); rand('state',100);
addpath(genpath('.'));

%% load mnist data
load mnist_part.mat; % with 10000 training & 10000 testing
% load mnist_all.mat; % the whole dataset
% binary the data set
rand('state',30);
[numdims,ntrain] = size(traindata); traindata = +(traindata>=rand(numdims,ntrain));
[numdims,ntest] = size(testdata); testdata = +(testdata>=rand(numdims,ntest));

%% SBN
K = 100; 

% sbn + gibbs
opts.burnin = 100; opts.space = 1; opts.sp = 100;
opts.interval = 1; opts.plotNow = 1;
result_sbngibbs = sbn_gibbs(traindata,testdata,K,opts);
% without pretraining
result_sbngibbs2 = dsbn_gibbs(traindata,testdata,K,K,opts);

% sbn + vb
opts.maxit = 100; opts.mcsamples = 1; opts.interval = 1; opts.plotNow = 1;
result_sbnvb = sbn_vb(traindata,testdata,K,opts);
% without pretraining
result_sbnvb2 = dsbn_vb(traindata,testdata,K,K,opts);

%% ARSBN
K = 100; 

% arsbn + gibbs
opts.burnin = 100; opts.space = 1; opts.sp = 100;
opts.interval = 1; opts.plotNow = 1;
result_fvsbngibbs = fvsbn_gibbs(traindata,testdata,opts);

% arsbn + vb
opts.maxit = 100; opts.mcsamples = 1; opts.interval = 1; opts.plotNow = 1;
result_fvsbnvb = fvsbn_vb(traindata,testdata,opts);
result_arsbnvb = arsbn_vb(traindata,testdata,K,opts);

% without pretraining
result_arsbnvb2 = arsbn2_vb(traindata,testdata,K,K,opts);

%% Multi-task SBN setup
K = 100; q = 10;
opts.maxit = 100; opts.mcsamples = 1; opts.interval = 1; opts.plotNow = 1;
result_multi1 = sbn_multitask_vb(traindata,testdata,trainlabel,testlabel,K,q,opts);
result_multi2 = dsbn_multitask_vb(traindata,testdata,trainlabel,testlabel,K,K,q,opts);
