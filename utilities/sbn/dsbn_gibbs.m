function result = dsbn_gibbs(Vtrain,Vtest,K1,K2,opts)
%% Bayesian Inference for Sigmoid Belief Network via Gibbs Sampling
% By Zhe Gan (zhe.gan@duke.edu), Duke ECE, 9.2.2014
% V = sigmoid(W1*H1+c1), H1 = sigmoid(W2*H2+c2), H2 = sigmoid(b)
% Input:
%       Vtrain: p*ntrain training data
%       Vtest:  p*ntest  test     data
%       K1,K2:  number of latent hidden units
%       opts:   parameters of Gibbs Sampling
% Output:
%       result: inferred matrix information

[p,ntrain] = size(Vtrain); [~,ntest] = size(Vtest);

%% initialize W and H
% if we obtain the pretraining results, then we do not need to initialize
% the model parameters randomly.
e0 = 1.1; f0 = 0.011;  phiW1 = ones(p,K1); phiW2 = ones(K1,K2);
W1 = 0.1*randn(p,K1); W2 = 0.1*randn(K1,K2); 
c1 = 0.1*randn(p,1); c2 = 0.1*randn(K1,1); b = zeros(K2,1);

prob = 1./(1+exp(-b));
H2train = +(repmat(prob,1,ntrain) > rand(K2,ntrain));  
H2test = +(repmat(prob,1,ntest) > rand(K2,ntest)); 
X = W2*H2train; prob = 1./(1+exp(-X)); H1train = +(prob>=rand(K1,ntrain));
X = W2*H2test; prob = 1./(1+exp(-X)); H1test = +(prob>=rand(K1,ntest));

%% initialize gibbs sampler parameters
iter = 0;
maxit = opts.burnin + opts.sp*opts.space;
TrainAcc = zeros(1,maxit); TestAcc = zeros(1,maxit);
TotalTime = zeros(1,maxit);
TrainLogProb = zeros(1,maxit); 
TestLogProb = zeros(1,maxit);

result.W1 = zeros(p,K1); result.W2 = zeros(K1,K2);
result.H1train = zeros(K1,ntrain); result.H2train = zeros(K2,ntrain);
result.H1test = zeros(K1,ntest); result.H2test = zeros(K2,ntest);
result.c1 = zeros(p,1);result.c2 = zeros(K1,1);result.b = zeros(K2,1); 
result.Vtrain = zeros(p,ntrain); result.Vtest = zeros(p,ntest);
result.gamma0Train = zeros(p,ntrain);
result.gamma0Test = zeros(p,ntest);
result.gamma1Train = zeros(K1,ntrain);
result.gamma1Test = zeros(K1,ntest);
result.gamma2 = zeros(K2,1);

%% Gibbs sampling
tic;
while (iter < maxit)
    iter = iter + 1;
    
    % 1. update gamma0
    Xmat = bsxfun(@plus,W1*H1train,c1); % p*n
    Xvec = reshape(Xmat,p*ntrain,1);
    gamma0vec = PolyaGamRndTruncated(ones(p*ntrain,1),Xvec,20);
    gamma0Train = reshape(gamma0vec,p,ntrain);
    
    Xmat = bsxfun(@plus,W1*H1test,c1); % p*n
    Xvec = reshape(Xmat,p*ntest,1);
    gamma0vec = PolyaGamRndTruncated(ones(p*ntest,1),Xvec,20);
    gamma0Test = reshape(gamma0vec,p,ntest);
    
    % 2. update W1
    jset=randperm(p);
    for j = jset        
        Hgam = bsxfun(@times,H1train,gamma0Train(j,:));
        invSigmaW = diag(phiW1(j,:)) + Hgam*H1train';
        MuW = invSigmaW\(sum(bsxfun(@times,H1train,Vtrain(j,:)-0.5-c1(j)*gamma0Train(j,:)),2));
        R = choll(invSigmaW); 
        W1(j,:) = (MuW + R\randn(K1,1))';
    end;
    
    % update phiW1
    % For simplicity, we use student't prior for W. 
    % The sampling of inverse Gaussian distribution is very slow, so 
    % TPBN shrinkage prior is only utilized in the VB solution.
    phiW1 = gamrnd(e0+0.5,1./(f0+0.5*W1.*W1));
    
    % 3. update H1
    res = W1*H1train;
    kset=randperm(K1);
    for k = kset
        res = res-W1(:,k)*H1train(k,:);
        mat1 = bsxfun(@plus,res,c1);
        vec1 = sum(bsxfun(@times,Vtrain-0.5-gamma0Train.*mat1,W1(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Train,W1(:,k).^2))/2; % 1*n
        logz = vec1 - vec2 + W2(k,:)*H2train+c2(k); % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        H1train(k,:) = (probz>rand(1,ntrain));
        res = res+W1(:,k)*H1train(k,:);
    end;
    
    res = W1*H1test;
    kset=randperm(K1);
    for k = kset
        res = res-W1(:,k)*H1test(k,:);
        mat1 = bsxfun(@plus,res,c1);
        vec1 = sum(bsxfun(@times,Vtest-0.5-gamma0Test.*mat1,W1(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Test,W1(:,k).^2))/2; % 1*n
        logz = vec1 - vec2 + W2(k,:)*H2test; % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        H1test(k,:) = (probz>rand(1,ntest));
        res = res+W1(:,k)*H1test(k,:);
    end;
    
    % 4. update c1
    sigmaC = 1./(sum(gamma0Train,2)+1);
    muC = sigmaC.*sum(Vtrain-0.5-gamma0Train.*(W1*H1train),2);
    c1 = normrnd(muC,sqrt(sigmaC));
    
    % 5. update gamma1
    Xmat = bsxfun(@plus,W2*H2train,c2); % p*n
    Xvec = reshape(Xmat,K1*ntrain,1);
    gamma1vec = PolyaGamRndTruncated(ones(K1*ntrain,1),Xvec,20);
    gamma1Train = reshape(gamma1vec,K1,ntrain);
    
    Xmat = bsxfun(@plus,W2*H2test,c2); % p*n
    Xvec = reshape(Xmat,K1*ntest,1);
    gamma1vec = PolyaGamRndTruncated(ones(K1*ntest,1),Xvec,20);
    gamma1Test = reshape(gamma1vec,K1,ntest);
    
    % 6. update W2
    kset=randperm(K1);
    for k = kset        
        Hgam = bsxfun(@times,H2train,gamma1Train(k,:)); % k2*n
        invSigmaW = diag(phiW2(k,:)) + Hgam*H2train'; % k2*k2
        MuW = invSigmaW\(sum(bsxfun(@times,H2train,H1train(k,:)-0.5-c2(k)*gamma1Train(k,:)),2)); % k2*1
        R = choll(invSigmaW); 
        W2(k,:) = (MuW + R\randn(K2,1))';
    end;
    
    % update phiW2
    phiW2 = gamrnd(e0+0.5,1./(f0+0.5*W2.*W2));
    
    % 7. update H2
    res = W2*H2train;
    kset=randperm(K2);
    for k = kset
        res = res-W2(:,k)*H2train(k,:); % k1*n
        mat1 = bsxfun(@plus,res,c2);
        vec1 = sum(bsxfun(@times,H1train-0.5-gamma1Train.*mat1,W2(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma1Train,W2(:,k).^2))/2; % 1*n
        logz = vec1 - vec2 + b(k); % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        H2train(k,:) = (probz>rand(1,ntrain));
        res = res+W2(:,k)*H2train(k,:); % k1*n
    end;
    
    res = W2*H2test;
    kset=randperm(K2);
    for k = kset
        res = res-W2(:,k)*H2test(k,:); % k1*n
        mat1 = bsxfun(@plus,res,c2);
        vec1 = sum(bsxfun(@times,H1test-0.5-gamma1Test.*mat1,W2(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma1Test,W2(:,k).^2))/2; % 1*n
        logz = vec1 - vec2 + b(k); % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        H2test(k,:) = (probz>rand(1,ntest));
        res = res+W2(:,k)*H2test(k,:); % k1*n
    end;
    
    % 8. update c2
    sigmaC = 1./(sum(gamma1Train,2)+1);
    muC = sigmaC.*sum(H1train-0.5-gamma1Train.*(W2*H2train),2);
    c2 = normrnd(muC,sqrt(sigmaC));

    % 9. update b
    gamma2 = PolyaGamRndExact(ones(K2,1),b);
    sigmaB = 1./(ntrain*gamma2+1e0);
    muB = sigmaB.*sum(H2train-0.5,2);
    b = normrnd(muB,sqrt(sigmaB));
    
    % 10. reconstruct the images
    X = bsxfun(@plus,W1*H1train,c1); % p*n
    prob = 1./(1+exp(-X));
    VtrainRecons = (prob>rand(p,ntrain));
    
    X = bsxfun(@plus,W1*H1test,c1); % p*n
    prob = 1./(1+exp(-X));
    VtestRecons = (prob>rand(p,ntest));
    
    % 11. Save samples.
    ndx = iter - opts.burnin;
    test = mod(ndx,opts.space);  
    if (ndx>0) && (test==0)
        result.W1 = result.W1 + W1/sp; result.W2 = result.W2 + W2/sp;
        result.H1train = result.H1train + H1train/sp; result.H2train = result.H2train + H2train/sp;
        result.H1test = result.H1test + H1test/sp; result.H2test = result.H2test + H2test/sp;
        result.b = result.b + b/sp;
        result.c1 = result.c1 + c1/sp;
        result.c2 = result.c2 + c2/sp;
        result.Vtrain = result.Vtrain + VtrainRecons/sp;
        result.Vtest = result.Vtest + VtestRecons/sp;
        result.gamma0Train = result.gamma0Train + gamma0Train/sp;
        result.gamma0Test = result.gamma0Test + gamma0Test/sp;
        result.gamma1Train = result.gamma1Train + gamma1Train/sp;
        result.gamma1Test = result.gamma1Test + gamma1Test/sp;
        result.gamma2 = result.gamma2 + gamma2/sp;
    end;
    
    TrainAcc(iter) = sum(sum(VtrainRecons==Vtrain))/p/ntrain;
    TestAcc(iter) = sum(sum(VtestRecons==Vtest))/p/ntest;
   
    % Note that, below is just the likelihood, not the marginal likelihood.
    % The marginal likelihood can be evaluated by using the
    % "functionEvaluation" file in the folder "support".
    
    % log prob. of training data
    mat = bsxfun(@plus,W1*H1train,c1);
    TrainLogProb(iter) = sum(sum(mat.*Vtrain-log(1+exp(mat))))/ntrain;
    % log prob. of test data
    mat = bsxfun(@plus,W1*H1test,c1);
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
            subplot(1,2,1); imagesc(W1); colorbar; title('W1');
            subplot(1,2,2); imagesc(W2); colorbar; title('W2');
            figure(3);
            dispims(W1,28,28); title('dictionaries');
            drawnow;
        end;
    end
end;
result.TrainAcc = TrainAcc; 
result.TestAcc = TestAcc;
result.TotalTime = TotalTime;
result.TrainLogProb = TrainLogProb; 
result.TestLogProb = TestLogProb;


