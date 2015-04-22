function result = sbn_gibbs(Vtrain,Vtest,K,opts)
%% Bayesian Inference for Sigmoid Belief Network via Gibbs Sampling
% By Zhe Gan (zhe.gan@duke.edu), Duke ECE, 9.1.2014
% V = sigmoid(W*H+c), H = sigmoid(b)
% Input:
%       Vtrain: p*ntrain training data
%       Vtest:  p*ntest  test     data
%       K:      number of latent hidden units
%       opts:   parameters of Gibbs Sampling
% Output:
%       result: inferred matrix information

[p,ntrain] = size(Vtrain); [~,ntest] = size(Vtest);

%% initialize W and H
b = 0.1*randn(K,1); c = 0.1*randn(p,1);
prob = 1./(1+exp(-b));
Htrain = +(repmat(prob,1,ntrain) > rand(K,ntrain));  
Htest = +(repmat(prob,1,ntest) > rand(K,ntest)); 
e0 = 1.1; f0 = 0.01; phiW = ones(p,K);
W = 0.1*randn(p,K);

% initialize gibbs sampler parameters
iter = 0;
maxit = opts.burnin + opts.sp*opts.space;
TrainAcc = zeros(1,maxit); TestAcc = zeros(1,maxit);
TotalTime = zeros(1,maxit);
TrainLogProb = zeros(1,maxit); 
TestLogProb = zeros(1,maxit);

result.W = zeros(p,K);
result.Htrain = zeros(K,ntrain);
result.Htest = zeros(K,ntest);
result.b = zeros(K,1);
result.c = zeros(p,1);
result.Vtrain = zeros(p,ntrain);
result.Vtest = zeros(p,ntest);
result.gamma0Train = zeros(p,ntrain);
result.gamma0Test = zeros(p,ntest);
result.gamma1 = zeros(K,1);

%% Gibbs sampling
tic;
while (iter < maxit)
    iter = iter + 1;

    % 1. update gamma0
    Xmat = bsxfun(@plus,W*Htrain,c); % p*n
    Xvec = reshape(Xmat,p*ntrain,1);
    gamma0vec = PolyaGamRndTruncated(ones(p*ntrain,1),Xvec,20);
    gamma0Train = reshape(gamma0vec,p,ntrain);
    
    Xmat = bsxfun(@plus,W*Htest,c); % p*n
    Xvec = reshape(Xmat,p*ntest,1);
    gamma0vec = PolyaGamRndTruncated(ones(p*ntest,1),Xvec,20);
    gamma0Test = reshape(gamma0vec,p,ntest);
    
    % 2. update W
    jset=randperm(p);
    for j = jset        
        Hgam = bsxfun(@times,Htrain,gamma0Train(j,:));
        invSigmaW = diag(phiW(j,:)) + Hgam*Htrain';
        MuW = invSigmaW\(sum(bsxfun(@times,Htrain,Vtrain(j,:)-0.5-c(j)*gamma0Train(j,:)),2));
        R = choll(invSigmaW); 
        W(j,:) = (MuW + R\randn(K,1))';
    end;
    
    % update gammaW
    % For simplicity, we use student't prior for W. 
    % The sampling of inverse Gaussian distribution is very slow, so 
    % TPBN shrinkage prior is only utilized in the VB solution.
    phiW = gamrnd(e0+0.5,1./(f0+0.5*W.*W));

    % 3. update H
    res = W*Htrain;
    kset=randperm(K);
    for k = kset
        res = res-W(:,k)*Htrain(k,:);
        mat1 = bsxfun(@plus,res,c);
        vec1 = sum(bsxfun(@times,Vtrain-0.5-gamma0Train.*mat1,W(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Train,W(:,k).^2))/2; % 1*n
        logz = vec1 - vec2 + b(k); % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        Htrain(k,:) = (probz>rand(1,ntrain));
        res = res+W(:,k)*Htrain(k,:);
    end;
    
    res = W*Htest;
    kset=randperm(K);
    for k = kset
        res = res-W(:,k)*Htest(k,:);
        mat1 = bsxfun(@plus,res,c);
        vec1 = sum(bsxfun(@times,Vtest-0.5-gamma0Test.*mat1,W(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Test,W(:,k).^2))/2; % 1*n
        logz = vec1 - vec2 + b(k); % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        Htest(k,:) = (probz>rand(1,ntest));
        res = res+W(:,k)*Htest(k,:);
    end;
    
    % 4. update c
    sigmaC = 1./(sum(gamma0Train,2)+1);
    muC = sigmaC.*sum(Vtrain-0.5-gamma0Train.*(W*Htrain),2);
    c = normrnd(muC,sqrt(sigmaC));

    % 5. update b
    gamma1 = PolyaGamRndExact(ones(K,1),b);
    sigmaB = 1./(ntrain*gamma1+1e0);
    muB = sigmaB.*sum(Htrain-0.5,2);
    b = normrnd(muB,sqrt(sigmaB));
    
    % 6. reconstruct the images
    X = bsxfun(@plus,W*Htrain,c); % p*n
    prob = 1./(1+exp(-X));
    VtrainRecons = (prob>rand(p,ntrain));
    
    X = bsxfun(@plus,W*Htest,c); % p*n
    prob = 1./(1+exp(-X));
    VtestRecons = (prob>rand(p,ntest));
    
    % 7. Save samples.
    ndx = iter - opts.burnin;
    test = mod(ndx,opts.space);  
    if (ndx>0) && (test==0)
        result.W = result.W + W/sp;
        result.Htrain = result.Htrain + Htrain/sp;
        result.Htest = result.Htest + Htest/sp;
        result.b = result.b + b/sp;
        result.c = result.c + c/sp;
        result.Vtrain = result.Vtrain + VtrainRecons/sp;
        result.Vtest = result.Vtest + VtestRecons/sp;
        result.gamma0Train = result.gamma0Train + gamma0Train/sp;
        result.gamma0Test = result.gamma0Test + gamma0Test/sp;
        result.gamma1 = result.gamma1 + gamma1/sp;
    end;
    
    TrainAcc(iter) = sum(sum(VtrainRecons==Vtrain))/p/ntrain;
    TestAcc(iter) = sum(sum(VtestRecons==Vtest))/p/ntest;
   
    % Note that, below is just the likelihood, not the marginal likelihood.
    % The marginal likelihood can be evaluated by using the
    % "functionEvaluation" file in the folder "support".
    
    % log prob. of training data
    mat = bsxfun(@plus,W*Htrain,c);
    TrainLogProb(iter) = sum(sum(mat.*Vtrain-log(1+exp(mat))))/ntrain;
    % log prob. of test data
    mat = bsxfun(@plus,W*Htest,c);
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
            subplot(1,2,1); imagesc(W); colorbar; title('W');
            subplot(1,2,2); imagesc(Htrain); colorbar; title('Htrain');
            figure(3);
            dispims(W,28,28); title('dictionaries');
            drawnow;
        end;
    end
end;
result.TrainAcc = TrainAcc; 
result.TestAcc = TestAcc;
result.TotalTime = TotalTime;
result.TrainLogProb = TrainLogProb; 
result.TestLogProb = TestLogProb;


