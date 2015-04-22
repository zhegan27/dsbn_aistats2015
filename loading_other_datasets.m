
% below we show how to import the other two datasets used in the
% experiments.

%% 1. ocr_letter
clear all;
load OCR_letter.mat;
traindata = data(1:42152,:)'; trainlabel = label(1:42152,:);
testdata = data(42153:end,:)'; testlabel = label(42153:end,:);
clear crossvalid data label;
index = randperm(42152);
figure; dispims0(traindata(:,index(1:100)),16,8,1); title('training samples');

%% 2. caltech101
clear all;
load caltech101_silhouettes_28_split1.mat;
traindata = [train_data;val_data]'; %trainlabel = [train_labels;val_labels];
testdata = test_data'; %testlabel = test_labels;
clear train_data train_labels val_data val_labels test_data test_labels;