function result = sbn_multitask_vb(Vtrain,Vtest,trainlabel,testlabel,K,q,opts)
%% Bayesian Inference for Sigmoid Belief Network via Variational Bayes
% By Zhe Gan (zhe.gan@duke.edu), Duke ECE, 10.11.2014
% V = sigmoid(W*H+c), H[l] = sigmoid(b[l])
% Input:
%       Vtrain: p*ntrain training data
%       Vtest:  p*ntest  test     data
%       K: number of latent hidden units
%       q: number of classes
%       opts: parameters of variational inference
% Output:
%       result: inferred matrix information

[p,ntrain] = size(Vtrain); [~,ntest] = size(Vtest);

%% initialize W and H
b = 0.1*randn(K,q); c = 0.1*randn(p,1);
Htrain = ones(K,ntrain); Htest = ones(K,ntest);
for i = 1:q
    prob = 1./(1+exp(-b(:,i)));
    num = sum(trainlabel==i-1);
    Htrain(:,trainlabel==i-1) = +(repmat(prob,1,num)>rand(K,num)); 
    num = sum(testlabel==i-1);
    Htest(:,testlabel==i-1) = +(repmat(prob,1,num)>rand(K,num)); 
end;
invgammaW = ones(p,K); xi = 1/2*ones(p,K); phi = ones(K,1); w = 1/2;
W = 0.1*randn(p,K); EWW = W.*W;

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
    
    % 1. update gamma0
    mat1 = bsxfun(@plus,W*Htrain,c);
    mat2 = (W.^2)*(Htrain.*(1-Htrain));
    mat3 = sqrt(mat1.^2 + mat2);
    gamma0Train = 1/2./(mat3+realmin).*tanh(mat3/2+realmin);

    mat1 = bsxfun(@plus,W*Htest,c);
    mat2 = (W.^2)*(Htest.*(1-Htest));
    mat3 = sqrt(mat1.^2 + mat2);
    gamma0Test = 1/2./(mat3+realmin).*tanh(mat3/2+realmin);

    % 2. update H
    res = W*Htrain; kset=randperm(K);
    for k = kset
        res = res - W(:,k)*Htrain(k,:);
        mat1 = bsxfun(@plus,res,c);
        vec1 = sum(bsxfun(@times,Vtrain-0.5-gamma0Train.*mat1,W(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Train,EWW(:,k)))/2; % 1*n
        for i = 1:q
            logz = vec1(trainlabel==i-1) - vec2(trainlabel==i-1) + b(k,i); % 1*n
            probz = 1./(1+exp(-logz)); % 1*n
            Htrain(k,trainlabel==i-1) = probz; 
        end;
        res = res + W(:,k)*Htrain(k,:);
    end;
    
    res = W*Htest; kset=randperm(K);
    for k = kset
        res = res - W(:,k)*Htest(k,:);
        mat1 = bsxfun(@plus,res,c);
        vec1 = sum(bsxfun(@times,Vtest-0.5-gamma0Test.*mat1,W(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Test,EWW(:,k)))/2; % 1*n
        for i = 1:q
            logz = vec1(testlabel==i-1) - vec2(testlabel==i-1) + b(k,i); % 1*n
            probz = 1./(1+exp(-logz)); % 1*n
            Htest(k,testlabel==i-1) = probz; 
        end;
        res = res + W(:,k)*Htest(k,:);
    end;

    % 3. update W
    SigmaW = 1./(gamma0Train*Htrain'+invgammaW);
    jset=randperm(p);
    for j = jset    
        Hgam = bsxfun(@times,Htrain,gamma0Train(j,:));
        HH = Hgam*Htrain'+diag(sum(Hgam.*(1-Htrain),2));
        invSigmaW = diag(invgammaW(j,:)) + HH;
        MuW = invSigmaW\(sum(bsxfun(@times,Htrain,Vtrain(j,:)-0.5-c(j)*gamma0Train(j,:)),2));
        W(j,:) = MuW';
    end;
    EWW = W.^2 + SigmaW;
    
    % (1). update gammaW
    a_GIG = 2*xi; b_GIG = EWW; sqrtab = sqrt(a_GIG.*b_GIG);
    gammaW = (sqrt(b_GIG).*besselk(1,sqrtab))./(sqrt(a_GIG).*besselk(0,sqrtab));
    invgammaW = (sqrt(a_GIG).*besselk(1,sqrtab))./(sqrt(b_GIG).*besselk(0,sqrtab));
    % (2). update xi
    a_xi = 1; b_xi = bsxfun(@plus,gammaW,phi');
    xi = a_xi./b_xi;
    % (3). update phi
    a_phi = 0.5+0.5*p; b_phi = w + sum(xi)';
    phi = a_phi./b_phi;
    % (4). update w
    w = (0.5+0.5*K)/(1+sum(phi));

    % 4. update c
    sigmaC = 1./(sum(gamma0Train,2)+1);
    c = sigmaC.*sum(Vtrain-0.5-gamma0Train.*(W*Htrain),2);
    
    % 5. update b
    gamma1 = zeros(K,q);
    for i = 1:q
        gamma1(:,i) = 1/2./(b(:,i)+realmin).*tanh(b(:,i)/2+realmin);
        num = sum(trainlabel==i-1);
        sigmaB = 1./(num*gamma1(:,i)+1);
        b(:,i) = sigmaB.* sum(Htrain(:,trainlabel==i-1)-0.5,2);
    end;
    
    % 6. reconstruct the images
    X = bsxfun(@plus,W*Htrain,c); % p*n
    prob = 1./(1+exp(-X));
    VtrainRecons = (prob>0.5);
    
    X = bsxfun(@plus,W*Htest,c); % p*n
    prob = 1./(1+exp(-X));
    VtestRecons = (prob>0.5);
    
    TrainAcc(iter) = sum(sum(VtrainRecons==Vtrain))/p/ntrain;
    TestAcc(iter) = sum(sum(VtestRecons==Vtest))/p/ntest;
    
    % 6. calculate lower bound
    totalP0 = zeros(1,num);
    for i = 1:num
        Hsamp = Htrain>=rand(K,ntrain);
        mat1 = bsxfun(@plus,W*Hsamp,c);
        totalP0(i) = sum(sum(mat1.*Vtrain-log(1+exp(mat1)))); 
    end;
    trainP0 = mean(totalP0)/ntrain;
    trainP1 = 0;
    for i = 1:q
        num = sum(trainlabel==i-1);
        mat1 = bsxfun(@times,Htrain(:,trainlabel==i-1),b(:,i));
        trainP1 = trainP1 + sum(sum(mat1))-num*sum(log(1+exp(b(:,i)))); 
    end;
    trainP1 = trainP1/ntrain;    
    trainQ1 = sum(sum(Htrain.*log(Htrain+realmin)+(1-Htrain).*log(1-Htrain+realmin))); trainQ1 = trainQ1/ntrain;
    TrainLogProb(iter) = trainP0+trainP1-trainQ1;
    
    totalP0 = zeros(1,num);
    for i = 1:num
        Hsamp = Htest>=rand(K,ntest);
        mat1 = bsxfun(@plus,W*Hsamp,c);
        totalP0(i) = sum(sum(mat1.*Vtest-log(1+exp(mat1)))); 
    end;
    testP0 = mean(totalP0)/ntest;
    testP1 = 0;
    for i = 1:q
        num = sum(testlabel==i-1);
        mat1 = bsxfun(@times,Htest(:,testlabel==i-1),b(:,i));
        testP1 = testP1 + sum(sum(mat1))-num*sum(log(1+exp(b(:,i)))); 
    end;
    testP1 = testP1/ntest;   
    testQ1 = sum(sum(Htest.*log(Htest+realmin)+(1-Htest).*log(1-Htest+realmin))); testQ1 = testQ1/ntest;
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
            subplot(1,2,1); imagesc(W); colorbar; title('W');
            subplot(1,2,2); imagesc(Htrain); colorbar; title('Htrain');
            figure(3);
            dispims(W,28,28); title('dictionaries');
            drawnow;
        end;
    end
end;

result.W = W; result.b = b; result.c = c;
result.Htrain = Htrain; result.Htest = Htest;
result.gamma0Train = gamma0Train;
result.gamma0Test = gamma0Test;
result.gamma1 = gamma1;
result.TrainAcc = TrainAcc; 
result.TestAcc = TestAcc;
result.TotalTime = TotalTime;
result.TrainLogProb = TrainLogProb; 
result.TestLogProb = TestLogProb;

