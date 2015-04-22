function result = arsbn2_vb(Vtrain,Vtest,K1,K2,opts)
%% Bayesian Inference for Sigmoid Belief Network via Variational Bayes
% By Zhe Gan (zhe.gan@duke.edu), Duke ECE, 10.12.2014
% V = sigmoid(W1*H1+S1*V+c1), H1 = sigmoid(W2*H2+S2*H1+c2), 
% H2 = sigmoid(U*H2+b)
% Input:
%       Vtrain: p*ntrain training data
%       Vtest:  p*ntest  test     data
%       K: number of latent hidden units
%       opts: parameters of variational inference
% Output:
%       result: inferred matrix information

[p,ntrain] = size(Vtrain); [~,ntest] = size(Vtest);

%% initialize W,U,S,b,c
c1 = 0.1*randn(p,1); c2 = 0.1*randn(K1,1); b = 0.1*randn(K2,1);
S1 = 0.1*randn(p,p); for j = 1:p, S1(j,j:end)=0; end;
S2 = 0.1*randn(K1,K1); for j = 1:K1, S2(j,j:end)=0; end;
U = 0.1*randn(K2,K2); for k = 1:K2, U(k,k:end)=0; end;
W1 = 0.1*randn(p,K1); EWW1 = W1.*W1; 
W2 = 0.1*randn(K1,K2); EWW2 = W2.*W2; 

% initialize H
prob = 1./(1+exp(-b)); 
H2train = +(repmat(prob,1,ntrain)>rand(K2,ntrain)); 
H2test = +(repmat(prob,1,ntest)>rand(K2,ntest));
X = W2*H2train; prob = 1./(1+exp(-X)); H1train = +(prob>=rand(K1,ntrain));
X = W2*H2test; prob = 1./(1+exp(-X)); H1test = +(prob>=rand(K1,ntest));

% initialize vb parameters
iter = 0; 
TrainAcc = zeros(1,opts.maxit); TestAcc = zeros(1,opts.maxit);
TotalTime = zeros(1,opts.maxit);
TrainLogProb = zeros(1,opts.maxit); 
TestLogProb = zeros(1,opts.maxit);

num = opts.mcsamples;

%% variational inference
tic;
while (iter < opts.maxit)
    iter = iter + 1;

    % 1(1). update gamma0
    mat1 = bsxfun(@plus,W1*H1train+S1*Vtrain,c1);
    mat2 = (W1.^2)*(H1train.*(1-H1train));
    mat3 = sqrt(mat1.^2 + mat2);
    gamma0Train = 1/2./(mat3+realmin).*tanh(mat3/2+realmin);
    
    mat1 = bsxfun(@plus,W1*H1test+S1*Vtest,c1);
    mat2 = (W1.^2)*(H1test.*(1-H1test));
    mat3 = sqrt(mat1.^2 + mat2);
    gamma0Test = 1/2./(mat3+realmin).*tanh(mat3/2+realmin);
    
    % 1(1). update gamma1
    mat1 = bsxfun(@plus,W2*H2train+S2*H1train,c2);
    mat2 = (W2.^2)*(H2train.*(1-H2train));
    mat3 = sqrt(mat1.^2 + mat2);
    gamma1Train = 1/2./(mat3+realmin).*tanh(mat3/2+realmin);
    
    mat1 = bsxfun(@plus,W2*H2test+S2*H1test,c2);
    mat2 = (W2.^2)*(H2test.*(1-H2test));
    mat3 = sqrt(mat1.^2 + mat2);
    gamma1Test = 1/2./(mat3+realmin).*tanh(mat3/2+realmin);
    
    % 1(2). update gamma2
    mat1 = bsxfun(@plus,U*H2train,b);
    mat2 = (U.^2)*(H2train.*(1-H2train));
    mat3 = sqrt(mat1.^2 + mat2);
    gamma2Train = 1/2./(mat3+realmin).*tanh(mat3/2+realmin);
    
    % 2(1). update H1
    res = W1*H1train; kset=randperm(K1);
    for k = kset
        res = res - W1(:,k)*H1train(k,:);
        mat1 = bsxfun(@plus,res+S1*Vtrain,c1);
        vec1 = sum(bsxfun(@times,Vtrain-0.5-gamma0Train.*mat1,W1(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Train,EWW1(:,k)))/2; % 1*n
        logz = vec1 - vec2 + W2(k,:)*H2train + S2(k,:)*H1train + c2(k); % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        H1train(k,:) = probz; 
        res = res + W1(:,k)*H1train(k,:);
    end;
    
    res = W1*H1test; kset=randperm(K1);
    for k = kset
        res = res - W1(:,k)*H1test(k,:);
        mat1 = bsxfun(@plus,res+S1*Vtest,c1);
        vec1 = sum(bsxfun(@times,Vtest-0.5-gamma0Test.*mat1,W1(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Test,EWW1(:,k)))/2; % 1*n
        logz = vec1 - vec2 + W2(k,:)*H2test + S2(k,:)*H1test + c2(k); % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        H1test(k,:) = probz; 
        res = res + W1(:,k)*H1test(k,:);
    end;
    
    % 2(2). update H2
    res = W2*H2train; kset=randperm(K2);
    for k = kset
        res = res - W2(:,k)*H2train(k,:);
        mat1 = bsxfun(@plus,res+S2*H1train,c2);
        vec1 = sum(bsxfun(@times,H1train-0.5-gamma1Train.*mat1,W2(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma1Train,EWW2(:,k)))/2; % 1*n
        logz = vec1 - vec2 + U(k,:)*H2train + b(k); % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        H2train(k,:) = probz; 
        res = res + W2(:,k)*H2train(k,:);
    end;
    
    res = W2*H2test; kset=randperm(K2);
    for k = kset
        res = res - W2(:,k)*H2test(k,:);
        mat1 = bsxfun(@plus,res+S2*H1test,c2);
        vec1 = sum(bsxfun(@times,H1test-0.5-gamma1Test.*mat1,W2(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma1Test,EWW2(:,k)))/2; % 1*n
        logz = vec1 - vec2 + U(k,:)*H2test + b(k); % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        H2test(k,:) = probz; 
        res = res + W2(:,k)*H2test(k,:);
    end;

    % 3(1). update W1
    SigmaW1 = 1./(gamma0Train*H1train'+1);
    jset=randperm(p);
    for j = jset    
        Hgam = bsxfun(@times,H1train,gamma0Train(j,:));
        HH = Hgam*H1train'+diag(sum(Hgam.*(1-H1train),2));
        invSigmaW = eye(K1) + HH;
        MuW = invSigmaW\(sum(bsxfun(@times,H1train,Vtrain(j,:)-0.5-...
            (S1(j,:)*Vtrain+c1(j)).*gamma0Train(j,:)),2));
        W1(j,:) = MuW';
    end;
    EWW1 = W1.^2 + SigmaW1;
    
    % 3(2). update W2
    SigmaW2 = 1./(gamma1Train*H2train'+1);
    jset=randperm(K1);
    for j = jset    
        Hgam = bsxfun(@times,H2train,gamma1Train(j,:));
        HH = Hgam*H2train'+diag(sum(Hgam.*(1-H2train),2));
        invSigmaW = eye(K2) + HH;
        MuW = invSigmaW\(sum(bsxfun(@times,H2train,H1train(j,:)-0.5-...
            (S2(j,:)*H1train+c2(j)).*gamma1Train(j,:)),2));
        W2(j,:) = MuW';
    end;
    EWW2 = W2.^2 + SigmaW2;
    
    % 4(1). update S1
    S1(1,:) = zeros(1,p);
    jset=randperm(p-1)+1;
    for j = jset        
        Vgam = bsxfun(@times,Vtrain(1:j-1,:),gamma0Train(j,:));
        invSigmaS = eye(j-1) + Vgam*Vtrain(1:j-1,:)';
        MuS = invSigmaS\(sum(bsxfun(@times,Vtrain(1:j-1,:),Vtrain(j,:)-0.5-...
            (W1(j,:)*H1train+c1(j)).*gamma0Train(j,:)),2));
        S1(j,1:j-1) = MuS';
        S1(j,j:end) = zeros(1,p-j+1);
    end;
    
    % 4(2). update S2
    S2(1,:) = zeros(1,K1);
    kset=randperm(K1-1)+1;
    for j = kset        
        Hgam = bsxfun(@times,H1train(1:j-1,:),gamma1Train(j,:));
        HH = Hgam*H1train(1:j-1,:)'+diag(sum(Hgam.*(1-H1train(1:j-1,:)),2));
        invSigmaS = eye(j-1) + HH;
        MuS = invSigmaS\(sum(bsxfun(@times,H1train(1:j-1,:),H1train(j,:)-0.5-...
            (W2(j,:)*H2train+c2(j)).*gamma1Train(j,:)),2));
        S2(j,1:j-1) = MuS';
        S2(j,j:end) = zeros(1,K1-j+1);
    end;
    
    % 5. update U
    U(1,:) = zeros(1,K2);
    kset=randperm(K2-1)+1;
    for j = kset     
        Hgam = bsxfun(@times,H2train(1:j-1,:),gamma2Train(j,:));
        HH = Hgam*H2train(1:j-1,:)'+diag(sum(Hgam.*(1-H2train(1:j-1,:)),2));
        invSigmaU = eye(j-1) + HH;
        MuU = invSigmaU\(sum(bsxfun(@times,H2train(1:j-1,:),H2train(j,:)-0.5-...
            b(j).*gamma2Train(j,:)),2));
        U(j,1:j-1) = MuU';
        U(j,j:end) = zeros(1,K2-j+1);
    end;
    
    % 6(1). update c1
    sigmaC1 = 1./(sum(gamma0Train,2)+1);
    c1 = sigmaC1.*sum(Vtrain-0.5-gamma0Train.*(W1*H1train+S1*Vtrain),2);
    
    % 6(2). update c2
    sigmaC2 = 1./(sum(gamma1Train,2)+1);
    c2 = sigmaC2.*sum(H1train-0.5-gamma1Train.*(W2*H2train+S2*H1train),2);
    
    % 6(3). update b
    sigmaB = 1./(sum(gamma2Train,2)+1);
    b = sigmaB.*sum(H2train-0.5-gamma2Train.*(U*H2train),2);
  
    % 7. reconstruct the images
    mat1 = bsxfun(@plus,W1*H1train+S1*Vtrain,c1); % p*n
    prob = 1./(1+exp(-mat1));
    VtrainRecons = (prob>0.5);
    
    mat1 = bsxfun(@plus,W1*H1test+S1*Vtest,c1); % p*n
    prob = 1./(1+exp(-mat1));
    VtestRecons = (prob>0.5);
    
    TrainAcc(iter) = sum(sum(VtrainRecons==Vtrain))/p/ntrain;
    TestAcc(iter) = sum(sum(VtestRecons==Vtest))/p/ntest;
    
    % 8. calculate lower bound
    totalP0 = zeros(1,num); totalP1 = zeros(1,num); totalP2 = zeros(1,num);
    for i = 1:num
        H1samp = H1train>=rand(K1,ntrain);
        mat1 = bsxfun(@plus,W1*H1samp+S1*Vtrain,c1);
        totalP0(i) = sum(sum(mat1.*Vtrain-log(1+exp(mat1)))); 
        H2samp = H2train>=rand(K2,ntrain);
        mat2 = bsxfun(@plus,W2*H2samp+S2*H1samp,c2);
        totalP1(i) = sum(sum(mat2.*H1samp-log(1+exp(mat2))));
        mat3 = bsxfun(@plus,U*H2samp,b);
        totalP2(i) = sum(sum(mat3.*H2samp-log(1+exp(mat3))));
    end;
    trainP0 = mean(totalP0)/ntrain; trainP1 = mean(totalP1)/ntrain; trainP2 = mean(totalP2)/ntrain; 
    trainQ1 = sum(sum(H1train.*log(H1train+1e-3)+(1-H1train).*log(1-H1train+1e-3))); trainQ1 = trainQ1/ntrain;
    trainQ2 = sum(sum(H2train.*log(H2train+1e-3)+(1-H2train).*log(1-H2train+1e-3))); trainQ2 = trainQ2/ntrain;
    TrainLogProb(iter) = trainP0+trainP1+trainP2-trainQ1-trainQ2;
    
    totalP0 = zeros(1,num); totalP1 = zeros(1,num); totalP2 = zeros(1,num);
    for i = 1:num
        H1samp = H1test>=rand(K1,ntest);
        mat1 = bsxfun(@plus,W1*H1samp+S1*Vtest,c1);
        totalP0(i) = sum(sum(mat1.*Vtest-log(1+exp(mat1)))); 
        H2samp = H2test>=rand(K2,ntest);
        mat2 = bsxfun(@plus,W2*H2samp+S2*H1samp,c2);
        totalP1(i) = sum(sum(mat2.*H1samp-log(1+exp(mat2))));
        mat3 = bsxfun(@plus,U*H2samp,b);
        totalP2(i) = sum(sum(mat3.*H2samp-log(1+exp(mat3))));
    end;
    testP0 = mean(totalP0)/ntest; testP1 = mean(totalP1)/ntest; testP2 = mean(totalP2)/ntest; 
    testQ1 = sum(sum(H1test.*log(H1test+1e-3)+(1-H1test).*log(1-H1test+1e-3))); testQ1 = testQ1/ntest;
    testQ2 = sum(sum(H2test.*log(H2test+1e-3)+(1-H2test).*log(1-H2test+1e-3))); testQ2 = testQ2/ntest;
    TestLogProb(iter) = testP0+testP1+testP2-testQ1-testQ2;
    
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
            drawnow;
        end;
    end
end;

result.W1 = W1; result.W2 = W2;
result.S1 = S1; result.S2 = S2; result.U = U;
result.H1train = H1train; result.H1test = H1test;
result.H2train = H2train; result.H2test = H2test;
result.b = b; result.c1 = c1; result.c2 = c2;

result.TrainAcc = TrainAcc; 
result.TestAcc = TestAcc;
result.TotalTime = TotalTime;
result.TrainLogProb = TrainLogProb; 
result.TestLogProb = TestLogProb;
    



