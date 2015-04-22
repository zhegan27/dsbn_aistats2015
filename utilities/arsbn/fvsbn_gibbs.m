function result = fvsbn_gibbs(Vtrain,Vtest,opts)
%% Bayesian Inference for Fully Visible Sigmoid Belief Network via Gibbs Sampling
% By Zhe Gan (zhe.gan@duke.edu), Duke ECE, 9.1.2014
% V = sigmoid(W*V+c)
% Input:
%       Vtrain: p*ntrain training data
%       Vtest:  p*ntest  test     data
%       opts:   parameters of Gibbs Sampling
% Output:
%       result: inferred matrix information

[p,ntrain] = size(Vtrain); [~,ntest] = size(Vtest);

%% initialize W and H
e0 = 1.1; f0 = 0.01;
c = 0.1*randn(p,1);
phiW = ones(p,p); W = 0.1*randn(p,p);
for j = 1:p, W(j,j:end)=0; end;

% initialize gibbs sampler parameters
iter = 0;
maxit = opts.burnin + opts.sp*opts.space;
TrainAcc = zeros(1,maxit); TestAcc = zeros(1,maxit);
TotalTime = zeros(1,maxit);
TrainLogProb = zeros(1,maxit); 
TestLogProb = zeros(1,maxit);

result.W = zeros(p,p);
result.c = zeros(p,1);
result.Vtrain = zeros(p,ntrain);
result.Vtest = zeros(p,ntest);
result.gamma0Train = zeros(p,ntrain);
result.gamma0Test = zeros(p,ntest);

%% Gibbs sampling
tic;
while (iter < maxit)
    iter = iter + 1;

    % 1. update gamma0
    Xmat = bsxfun(@plus,W*Vtrain,c); % p*n
    Xvec = reshape(Xmat,p*ntrain,1);
    gamma0vec = PolyaGamRndTruncated(ones(p*ntrain,1),Xvec,20);
    gamma0Train = reshape(gamma0vec,p,ntrain);
    
    Xmat = bsxfun(@plus,W*Vtest,c); % p*n
    Xvec = reshape(Xmat,p*ntest,1);
    gamma0vec = PolyaGamRndTruncated(ones(p*ntest,1),Xvec,20);
    gamma0Test = reshape(gamma0vec,p,ntest);
    
    % 2. update W
    W(1,:) = zeros(1,p);
    jset=randperm(p-1)+1;
    for j = jset       
        Vgam = bsxfun(@times,Vtrain(1:j-1,:),gamma0Train(j,:));
        invSigmaW = diag(phiW(j,1:j-1)) + Vgam*Vtrain(1:j-1,:)';
        MuW = invSigmaW\(sum(bsxfun(@times,Vtrain(1:j-1,:),Vtrain(j,:)-0.5-c(j)*gamma0Train(j,:)),2));
        R = choll(invSigmaW); 
        W(j,1:j-1) = (MuW + R\randn(j-1,1))';
        W(j,j:end) = zeros(1,p-j+1);
    end;
    
    % update gammaW
    % For simplicity, we use student't prior for W. 
    % The sampling of inverse Gaussian distribution is very slow, so 
    % TPBN shrinkage prior is only utilized in the VB solution.
    phiW = gamrnd(e0+0.5,1./(f0+0.5*W.*W));

    % 4. update c
    sigmaC = 1./(sum(gamma0Train,2)+1);
    muC = sigmaC.*sum(Vtrain-0.5-gamma0Train.*(W*Vtrain),2);
    c = normrnd(muC,sqrt(sigmaC));
    
    % 6. reconstruct the images
    X = bsxfun(@plus,W*Vtrain,c); % p*n
    prob = 1./(1+exp(-X));
    VtrainRecons = (prob>rand(p,ntrain));
    
    X = bsxfun(@plus,W*Vtest,c); % p*n
    prob = 1./(1+exp(-X));
    VtestRecons = (prob>rand(p,ntest));
    
    % 7. Save samples.
    ndx = iter - opts.burnin;
    test = mod(ndx,opts.space);  
    if (ndx>0) && (test==0)
        result.W = result.W + W/sp;
        result.c = result.c + c/sp;
        result.Vtrain = result.Vtrain + VtrainRecons/sp;
        result.Vtest = result.Vtest + VtestRecons/sp;
        result.gamma0Train = result.gamma0Train + gamma0Train/sp;
        result.gamma0Test = result.gamma0Test + gamma0Test/sp;
    end;
    
    TrainAcc(iter) = sum(sum(VtrainRecons==Vtrain))/p/ntrain;
    TestAcc(iter) = sum(sum(VtestRecons==Vtest))/p/ntest;
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
result.TrainAcc = TrainAcc; 
result.TestAcc = TestAcc;
result.TotalTime = TotalTime;
result.TrainLogProb = TrainLogProb; 
result.TestLogProb = TestLogProb;


