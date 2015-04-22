function result = arsbn_vb(Vtrain,Vtest,K,opts)
%% Bayesian Inference for Sigmoid Belief Network via Variational Bayes
% By Zhe Gan (zhe.gan@duke.edu), Duke ECE, 10.12.2014
% V = sigmoid(W*H+S*V+c), H = sigmoid(U*H+b)
% Input:
%       Vtrain: p*ntrain training data
%       Vtest:  p*ntest  test     data
%       K: number of latent hidden units
%       opts: parameters of variational inference
% Output:
%       result: inferred matrix information

[p,ntrain] = size(Vtrain); [~,ntest] = size(Vtest);

%% initialize W,U,S,b,c
c = 0.1*randn(p,1);  b = 0.1*randn(K,1);
S = 0.1*randn(p,p); for j = 1:p, S(j,j:end)=0; end;
U = 0.1*randn(K,K); for k = 1:K, U(k,k:end)=0; end;

gammaW = ones(p,K); invgammaW = 10*ones(p,K); 
xi = 1/2*ones(p,K); phi = ones(K,1); w = 1/2;
W = randn(p,K); EWW = W.*W;

% initialize H
prob = 1./(1+exp(-b)); 
Htrain = +(repmat(prob,1,ntrain)>rand(K,ntrain)); 
Htest = +(repmat(prob,1,ntest)>rand(K,ntest));

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
    mat1 = bsxfun(@plus,W*Htrain+S*Vtrain,c);
    mat2 = (W.^2)*(Htrain.*(1-Htrain));
    mat3 = sqrt(mat1.^2 + mat2);
    gamma0Train = 1/2./(mat3+realmin).*tanh(mat3/2+realmin);
    
    mat1 = bsxfun(@plus,W*Htest+S*Vtest,c);
    mat2 = (W.^2)*(Htest.*(1-Htest));
    mat3 = sqrt(mat1.^2 + mat2);
    gamma0Test = 1/2./(mat3+realmin).*tanh(mat3/2+realmin);
    
    % 1(2). update gamma1
    mat1 = bsxfun(@plus,U*Htrain,b);
    mat2 = (U.^2)*(Htrain.*(1-Htrain));
    mat3 = sqrt(mat1.^2 + mat2);
    gamma1Train = 1/2./(mat3+realmin).*tanh(mat3/2+realmin);
    
    mat1 = bsxfun(@plus,U*Htest,b);
    mat2 = (U.^2)*(Htest.*(1-Htest));
    mat3 = sqrt(mat1.^2 + mat2);
    gamma1Test = 1/2./(mat3+realmin).*tanh(mat3/2+realmin);
    
    % 2. update H
    res = W*Htrain; kset=randperm(K);
    for k = kset
        res = res - W(:,k)*Htrain(k,:);
        mat1 = bsxfun(@plus,res+S*Vtrain,c);
        vec1 = sum(bsxfun(@times,Vtrain-0.5-gamma0Train.*mat1,W(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Train,EWW(:,k)))/2; % 1*n
        logz = vec1 - vec2 + U(k,:)*Htrain + b(k); % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        Htrain(k,:) = probz; 
        res = res + W(:,k)*Htrain(k,:);
    end;
    
    res = W*Htest; kset=randperm(K);
    for k = kset
        res = res - W(:,k)*Htest(k,:);
        mat1 = bsxfun(@plus,res+S*Vtest,c);
        vec1 = sum(bsxfun(@times,Vtest-0.5-gamma0Test.*mat1,W(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Test,EWW(:,k)))/2; % 1*n
        logz = vec1 - vec2 + U(k,:)*Htest + b(k,:); % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        Htest(k,:) = probz; 
        res = res + W(:,k)*Htest(k,:);
    end; 
    
    % 3. update W
    SigmaW = 1./(gamma0Train*Htrain'+invgammaW);
    jset=randperm(p);
    for j = jset   
        Hgam = bsxfun(@times,Htrain,gamma0Train(j,:));
        HH = Hgam*Htrain'+diag(sum(Hgam.*(1-Htrain),2));
        invSigmaW = diag(invgammaW(j,:)) + HH;
        MuW = invSigmaW\(sum(bsxfun(@times,Htrain,Vtrain(j,:)-0.5-...
            (S(j,:)*Vtrain+c(j)).*gamma0Train(j,:)),2));
        W(j,:) = MuW';
    end;
    EWW = W.^2 + SigmaW;

    % 4. update S
    S(1,:) = zeros(1,p);
    jset=randperm(p-1)+1;
    for j = jset        
        Vgam = bsxfun(@times,Vtrain(1:j-1,:),gamma0Train(j,:));
        invSigmaS = 10*eye(j-1) + Vgam*Vtrain(1:j-1,:)';
        MuS = invSigmaS\(sum(bsxfun(@times,Vtrain(1:j-1,:),Vtrain(j,:)-0.5-...
            (W(j,:)*Htrain+c(j)).*gamma0Train(j,:)),2));
        S(j,1:j-1) = MuS';
        S(j,j:end) = zeros(1,p-j+1);
    end;
    
    % 5. update U
    U(1,:) = zeros(1,K);
    kset=randperm(K-1)+1;
    for j = kset      
        Hgam = bsxfun(@times,Htrain(1:j-1,:),gamma1Train(j,:));
        HH = Hgam*Htrain(1:j-1,:)'+diag(sum(Hgam.*(1-Htrain(1:j-1,:)),2));
        invSigmaU = 10*eye(j-1) + HH;
        MuU = invSigmaU\(sum(bsxfun(@times,Htrain(1:j-1,:),Htrain(j,:)-0.5-...
            b(j).*gamma1Train(j,:)),2));
        U(j,1:j-1) = MuU';
        U(j,j:end) = zeros(1,K-j+1);
    end;
    
    % 6(1). update c
    sigmaC = 1./(sum(gamma0Train,2)+1);
    c = sigmaC.*sum(Vtrain-0.5-gamma0Train.*(W*Htrain+S*Vtrain),2);
    
    % 6(2). update b
    sigmaB = 1./(sum(gamma1Train,2)+1);
    b = sigmaB.*sum(Htrain-0.5-gamma1Train.*(U*Htrain),2);
  
    % 7. reconstruct the images
    mat1 = bsxfun(@plus,W*Htrain+S*Vtrain,c); % p*n
    prob = 1./(1+exp(-mat1));
    VtrainRecons = (prob>0.5);
    
    mat1 = bsxfun(@plus,W*Htest+S*Vtest,c); % p*n
    prob = 1./(1+exp(-mat1));
    VtestRecons = (prob>0.5);
    
    TrainAcc(iter) = sum(sum(VtrainRecons==Vtrain))/p/ntrain;
    TestAcc(iter) = sum(sum(VtestRecons==Vtest))/p/ntest;
    
    % 8. calculate lower bound
    totalP0 = zeros(1,num); totalP1 = zeros(1,num);
    for i = 1:num
        Hsamp = Htrain>=rand(K,ntrain);
        mat1 = bsxfun(@plus,W*Hsamp+S*Vtrain,c);
        totalP0(i) = sum(sum(mat1.*Vtrain-log(1+exp(mat1)))); 
        mat2 = bsxfun(@plus,U*Hsamp,b);
        totalP1(i) = sum(sum(mat2.*Hsamp-log(1+exp(mat2))));
    end;
    trainP0 = mean(totalP0)/ntrain; trainP1 = mean(totalP1)/ntrain;   
    trainQ1 = sum(sum(Htrain.*log(Htrain+1e-3)+(1-Htrain).*log(1-Htrain+1e-3))); trainQ1 = trainQ1/ntrain;
    TrainLogProb(iter) = trainP0+trainP1-trainQ1;
    
    totalP0 = zeros(1,num); totalP1 = zeros(1,num);
    for i = 1:num
        Hsamp = Htest>=rand(K,ntest);
        mat1 = bsxfun(@plus,W*Hsamp+S*Vtest,c);
        totalP0(i) = sum(sum(mat1.*Vtest-log(1+exp(mat1)))); 
        mat2 = bsxfun(@plus,U*Hsamp,b);
        totalP1(i) = sum(sum(mat2.*Hsamp-log(1+exp(mat2))));
    end;
    testP0 = mean(totalP0)/ntest; testP1 = mean(totalP1)/ntest;    
    testQ1 = sum(sum(Htest.*log(Htest+1e-3)+(1-Htest).*log(1-Htest+1e-3))); testQ1 = testQ1/ntest;
    TestLogProb(iter) = testP0+testP1-testQ1;
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
            subplot(2,2,1); imagesc(W); colorbar; title('W');
            subplot(2,2,2); imagesc(S); colorbar; title('S');
            subplot(2,2,3); imagesc(U); colorbar; title('U');
            figure(3);
            dispims(W,28,28); title('dictionaries');
            drawnow;
        end;
    end
end;

result.W = W; result.S = S; result.U = U;
result.Htrain = Htrain; result.Htest = Htest;
result.b = b; result.c = c;
result.gamma0Train = gamma0Train;
result.gamma0Test = gamma0Test;
result.gamma1Train = gamma1Train;
result.gamma1Test = gamma1Test;
result.TrainAcc = TrainAcc; 
result.TestAcc = TestAcc;
result.TotalTime = TotalTime;
result.TrainLogProb = TrainLogProb; 
result.TestLogProb = TestLogProb;

