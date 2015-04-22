function result = fvsbn_vb(Vtrain,Vtest,opts)
%% Bayesian Inference for Fully Visible Sigmoid Belief Network via Variational Bayes
% By Zhe Gan (zhe.gan@duke.edu), Duke ECE, 10.12.2014
% V = sigmoid(W*V+c)
% Input:
%       Vtrain: p*ntrain training data
%       Vtest:  p*ntest  test     data
%       opts:   parameters of variational inference
% Output:
%       result: inferred matrix information

[p,ntrain] = size(Vtrain); [~,ntest] = size(Vtest);

%% initialize W and H
c = 0.1*randn(p,1);
gammaW = ones(p,p); invgammaW = 10*ones(p,p); 
xi = 1/2*ones(p,p); phi = ones(p,1); w = 1/2;

W = 0.1*randn(p,p); for j = 1:p, W(j,j:end)=0; end;
SigmaW = zeros(p,p);

% initialize vb parameters
iter = 0; 
TrainAcc = zeros(1,opts.maxit); TestAcc = zeros(1,opts.maxit);
TotalTime = zeros(1,opts.maxit);
TrainLogProb = zeros(1,opts.maxit); 
TestLogProb = zeros(1,opts.maxit);

%% variational inference
tic;
while (iter < opts.maxit)
    iter = iter + 1;
    
    % 1. update gamma0
    X = bsxfun(@plus,W*Vtrain,c);
    gamma0Train = 1/2./(X+realmin).*tanh(X/2+realmin);
    
    X = bsxfun(@plus,W*Vtest,c);
    gamma0Test = 1/2./(X+realmin).*tanh(X/2+realmin);
    
    % 2. update W
    W(1,:) = zeros(1,p); SigmaW(1,:) = zeros(1,p);
    jset=randperm(p-1)+1;
    for j = jset        
        Vgam = bsxfun(@times,Vtrain(1:j-1,:),gamma0Train(j,:));
        invSigmaW = diag(invgammaW(j,1:j-1)) + Vgam*Vtrain(1:j-1,:)';
        MuW = invSigmaW\(sum(bsxfun(@times,Vtrain(1:j-1,:),Vtrain(j,:)-0.5-c(j)*gamma0Train(j,:)),2));
        W(j,1:j-1) = MuW';
        W(j,j:end) = zeros(1,p-j+1);
        SigmaW(j,1:j-1) = 1./(gamma0Train(j,:)*(Vtrain(1:j-1,:).^2)'+invgammaW(j,1:j-1));
        SigmaW(j,j:end) = zeros(1,p-j+1);
    end;
    
    % we use a gaussian prior on W here, no shrinkage is imposed, but can 
    % be easily employed.
    
    % 3. update c
    sigmaC = 1./(sum(gamma0Train,2)+1);
    c = sigmaC.*sum(Vtrain-0.5-gamma0Train.*(W*Vtrain),2);
    
    % 4. reconstruct the images
    X = bsxfun(@plus,W*Vtrain,c); % p*n
    prob = 1./(1+exp(-X));
    VtrainRecons = (prob>0.5);
    
    X = bsxfun(@plus,W*Vtest,c); % p*n
    prob = 1./(1+exp(-X));
    VtestRecons = (prob>0.5);
    
    TrainAcc(iter) = sum(sum(VtrainRecons==Vtrain))/p/ntrain;
    TestAcc(iter) = sum(sum(VtestRecons==Vtest))/p/ntest;
    
    % 5. calculate lower bound
    % log prob. of training data
    mat = bsxfun(@plus,W*Vtrain,c);
    TrainLogProb(iter)  = sum(sum(mat.*Vtrain-log(1+exp(mat))))/ntrain;
    % log prob. of test data
    mat = bsxfun(@plus,W*Vtest,c);
    TestLogProb(iter) = sum(sum(mat.*Vtest-log(1+exp(mat))))/ntest;
    
    TotalTime(iter) = toc;
    
    if mod(iter,opts.interval)==0
        disp(['Iteration: ' num2str(iter) ' Acc: ' num2str(TrainAcc(iter)) ' ' num2str(TestAcc(iter))...
            ' LogProb: ' num2str(TrainLogProb(iter))  ' ' num2str(TestLogProb(iter))...
             ' Totally spend ' num2str(TotalTime(iter))]);
         
        if  opts.plotNow == 1
            index = randperm(ntrain);
            figure(1);
            dispims(VtrainRecons(:,index(1:100)),28,28); title('Reconstruction');
            figure(2);
            imagesc(W); colorbar; title('W');
            drawnow;
        end;
    end
end;

result.W = W;
result.c = c;
result.gamma0Train = gamma0Train;
result.gamma0Test = gamma0Test;
result.TrainAcc = TrainAcc; 
result.TestAcc = TestAcc;
result.TotalTime = TotalTime;
result.TrainLogProb = TrainLogProb; 
result.TestLogProb = TestLogProb;

