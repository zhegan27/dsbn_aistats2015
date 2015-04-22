function result = dsbn_multitask_vb(Vtrain,Vtest,trainlabel,testlabel,...
    K1,K2,q,opts)
%% Bayesian Inference for Sigmoid Belief Network via Variational Bayes
% By Zhe Gan (zhe.gan@duke.edu), Duke ECE, 10.11.2014
% V = sigmoid(W1*H1+c1), H1 = sigmoid(W2*H2+c2), H2[l] = sigmoid(b[l])
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
invgammaW1 = ones(p,K1); xiW1 = 1/2*ones(p,K1); phiW1 = ones(K1,1); omegaW1 = 1/2;
W1 = 0.1*randn(p,K1); EWW1 = W1.*W1;

invgammaW2 = ones(K1,K2); xiW2 = 1/2*ones(K1,K2); phiW2 = ones(K2,1); omegaW2 = 1/2;
W2 = 0.1*randn(K1,K2); EWW2 = W2.*W2;

c1 = 0.1*randn(p,1); c2 = 0.1*randn(K1,1); b = 0.1*randn(K2,q);
H2train = ones(K2,ntrain); H2test = ones(K2,ntest);
for i = 1:q
    prob = 1./(1+exp(-b(:,i)));
    num = sum(trainlabel==i-1);
    H2train(:,trainlabel==i-1) = +(repmat(prob,1,num)>rand(K2,num)); 
    num = sum(testlabel==i-1);
    H2test(:,testlabel==i-1) = +(repmat(prob,1,num)>rand(K2,num)); 
end;
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
    
    % 1. update gamma0
    mat1 = bsxfun(@plus,W1*H1train,c1);
    mat2 = (W1.^2)*(H1train.*(1-H1train));
    mat3 = sqrt(mat1.^2 + mat2);
    gamma0Train = 1/2./(mat3+realmin).*tanh(mat3/2+realmin);

    mat1 = bsxfun(@plus,W1*H1test,c1);
    mat2 = (W1.^2)*(H1test.*(1-H1test));
    mat3 = sqrt(mat1.^2 + mat2);
    gamma0Test = 1/2./(mat3+realmin).*tanh(mat3/2+realmin);
    
    % 2. update H1
    res = W1*H1train; kset=randperm(K1);
    for k = kset
        res = res-W1(:,k)*H1train(k,:);
		mat1 = bsxfun(@plus,res,c1);
        vec1 = sum(bsxfun(@times,Vtrain-0.5-gamma0Train.*mat1,W1(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Train,EWW1(:,k)))/2; % 1*n
        logz = vec1 - vec2 + W2(k,:)*H2train+c2(k); % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        H1train(k,:) = probz; 
        res = res+W1(:,k)*H1train(k,:);
    end;
    
    res = W1*H1test; kset=randperm(K1);
    for k = kset
        res = res-W1(:,k)*H1test(k,:);
		mat1 = bsxfun(@plus,res,c1);
        vec1 = sum(bsxfun(@times,Vtest-0.5-gamma0Test.*mat1,W1(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Test,EWW1(:,k)))/2; % 1*n
        logz = vec1 - vec2 + W2(k,:)*H2test+c2(k); % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        H1test(k,:) = probz; 
        res = res+W1(:,k)*H1test(k,:);
    end;
    
    % 3. update W1
    SigmaW = 1./(gamma0Train*H1train'+invgammaW1);
    jset=randperm(p);
    for j = jset        
        Hgam = bsxfun(@times,H1train,gamma0Train(j,:));
        HH = Hgam*H1train'+diag(sum(Hgam.*(1-H1train),2));
        invSigmaW = diag(invgammaW1(j,:)) + HH;
        MuW = invSigmaW\(sum(bsxfun(@times,H1train,Vtrain(j,:)-0.5-c1(j)*gamma0Train(j,:)),2));
        W1(j,:) = MuW';
    end;
    EWW1 = W1.^2 + SigmaW;
    
    % Another way to update W
%     SigmaW = 1./(gamma0Train*H1train'+invgammaW1);
%     res = W1*H1train; kset=randperm(K1);
%     for k = kset
%         res = res-W1(:,k)*H1train(k,:);
%         mat = bsxfun(@plus,res,c1);
%         vec = (Vtrain-mat.*gamma0Train-1/2)*H1train(k,:)';
%         W1(:,k) = SigmaW(:,k).*vec;
%         res = res+W1(:,k)*H1train(k,:);
%     end;
%     EWW1 = W1.^2 + SigmaW;
    
    % (1). update gammaW1
    a_GIG = 2*xiW1; b_GIG = EWW1; sqrtab = sqrt(a_GIG.*b_GIG);
    gammaW1 = (sqrt(b_GIG).*besselk(1,sqrtab))./(sqrt(a_GIG).*besselk(0,sqrtab));
    invgammaW1 = (sqrt(a_GIG).*besselk(1,sqrtab))./(sqrt(b_GIG).*besselk(0,sqrtab));
    % (2). update xi
    a_xi = 1; b_xi = bsxfun(@plus,gammaW1,phiW1');
    xiW1 = a_xi./b_xi;
    % (3). update phi
    a_phi = 0.5+0.5*p; b_phi = omegaW1 + sum(xiW1)';
    phiW1 = a_phi./b_phi;
    % (4). update w
    omegaW1 = (0.5+0.5*K1)/(1+sum(phiW1));

	% 4. update c1
    sigmaC = 1./(sum(gamma0Train,2)+1);
    c1 = sigmaC.*sum(Vtrain-0.5-gamma0Train.*(W1*H1train),2);
    
    % 5. update gamma1    
    mat1 = bsxfun(@plus,W2*H2train,c2);
    mat2 = (W2.^2)*(H2train.*(1-H2train));
    mat3 = sqrt(mat1.^2 + mat2);
    gamma1Train = 1/2./(mat3+realmin).*tanh(mat3/2+realmin);

    mat1 = bsxfun(@plus,W2*H2test,c2);
    mat2 = (W2.^2)*(H2test.*(1-H2test));
    mat3 = sqrt(mat1.^2 + mat2);
    gamma1Test = 1/2./(mat3+realmin).*tanh(mat3/2+realmin);
    
    % 6. update H2
    res = W2*H2train; kset=randperm(K2);
    for k = kset
        res = res-W2(:,k)*H2train(k,:); % k1*n
		mat1 = bsxfun(@plus,res,c2);
        vec1 = sum(bsxfun(@times,H1train-0.5-gamma1Train.*mat1,W2(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma1Train,EWW2(:,k)))/2; % 1*n
        for i = 1:q
            logz = vec1(trainlabel==i-1) - vec2(trainlabel==i-1) + b(k,i); % 1*n
            probz = 1./(1+exp(-logz)); % 1*n
            H2train(k,trainlabel==i-1) = probz; 
        end;
        res = res+W2(:,k)*H2train(k,:); % k1*n
    end;
    
    res = W2*H2test; kset=randperm(K2);
    for k = kset
        res = res-W2(:,k)*H2test(k,:); % k1*n
		mat1 = bsxfun(@plus,res,c2);
        vec1 = sum(bsxfun(@times,H1test-0.5-gamma1Test.*mat1,W2(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma1Test,EWW2(:,k)))/2; % 1*n
        for i = 1:q
            logz = vec1(testlabel==i-1) - vec2(testlabel==i-1) + b(k,i); % 1*n
            probz = 1./(1+exp(-logz)); % 1*n
            H2test(k,testlabel==i-1) = probz; 
        end;
        res = res+W2(:,k)*H2test(k,:); % k1*n
    end;
    
    % 7. update W2
    SigmaW = 1./(gamma1Train*H2train'+invgammaW2);
    kset=randperm(K1);
    for j = kset        
        Hgam = bsxfun(@times,H2train,gamma1Train(j,:));
        HH = Hgam*H2train'+diag(sum(Hgam.*(1-H2train),2));
        invSigmaW = diag(invgammaW2(j,:)) + HH;
        MuW = invSigmaW\(sum(bsxfun(@times,H2train,H1train(j,:)-0.5-c2(j)*gamma1Train(j,:)),2));
        W2(j,:) = MuW';
    end;
    EWW2 = W2.^2 + SigmaW;

    % Another way to update W
%     SigmaW = 1./(gamma1Train*H2train'+invgammaW2);
%     res = W2*H2train; kset=randperm(K2);
%     for k = kset
%         res = res-W2(:,k)*H2train(k,:);
%         mat = bsxfun(@plus,res,c2);
%         vec = (H1train-mat.*gamma1Train-1/2)*H2train(k,:)';
%         W2(:,k) = SigmaW(:,k).*vec;
%         res = res+W2(:,k)*H2train(k,:);
%     end;
%     EWW2 = W2.^2 + SigmaW;
    
    % (1). update gammaW2
    a_GIG = 2*xiW2; b_GIG = EWW2; sqrtab = sqrt(a_GIG.*b_GIG);
    gammaW2 = (sqrt(b_GIG).*besselk(1,sqrtab))./(sqrt(a_GIG).*besselk(0,sqrtab));
    invgammaW2 = (sqrt(a_GIG).*besselk(1,sqrtab))./(sqrt(b_GIG).*besselk(0,sqrtab));
    % (2). update xi
    a_xi = 1; b_xi = bsxfun(@plus,gammaW2,phiW2');
    xiW2 = a_xi./b_xi;
    % (3). update phi
    a_phi = 0.5+0.5*K1; b_phi = omegaW2 + sum(xiW2)';
    phiW2 = a_phi./b_phi;
    % (4). update w
    omegaW2 = (0.5+0.5*K2)/(1+sum(phiW2));

	% 8. update c2
    sigmaC = 1./(sum(gamma1Train,2)+1);
    c2 = sigmaC.*sum(H1train-0.5-gamma1Train.*(W2*H2train),2);

    % 9. update b
    gamma2 = zeros(K2,q);
    for i = 1:q
        gamma2(:,i) = 1/2./(b(:,i)+realmin).*tanh(b(:,i)/2+realmin);
        num = sum(trainlabel==i-1);
        sigmaB = 1./(num*gamma2(:,i)+1);
        b(:,i) = sigmaB.* sum(H2train(:,trainlabel==i-1)-0.5,2);
    end;
    
    % 10. reconstruct the images
    X = bsxfun(@plus,W1*H1train,c1); % p*n
    prob = 1./(1+exp(-X));
    VtrainRecons = (prob>0.5);
    
    X = bsxfun(@plus,W1*H1test,c1); % p*n
    prob = 1./(1+exp(-X));
    VtestRecons = (prob>0.5);
 
    TrainAcc(iter) = sum(sum(VtrainRecons==Vtrain))/p/ntrain;
    TestAcc(iter) = sum(sum(VtestRecons==Vtest))/p/ntest;
    
    % 11. calculate lower bound
    totalP0 = zeros(1,num); totalP1 = zeros(1,num);
    for i = 1:num
        H1samp = H1train>=rand(K1,ntrain);
        mat1 = bsxfun(@plus,W1*H1samp,c1);
        totalP0(i) = sum(sum(mat1.*Vtrain-log(1+exp(mat1)))); 
        H2samp = H2train>=rand(K2,ntrain);
        mat2 = bsxfun(@plus,W2*H2samp,c2);
        totalP1(i) = sum(sum(mat2.*H1train-log(1+exp(mat2)))); 
    end;
    trainP0 = mean(totalP0)/ntrain; trainP1 = mean(totalP1)/ntrain;
    trainQ1 = sum(sum(H1train.*log(H1train+realmin)+(1-H1train).*log(1-H1train+realmin))); trainQ1 = trainQ1/ntrain;
    trainP2 = 0;
    for i = 1:q
        num = sum(trainlabel==i-1);
        mat1 = bsxfun(@times,H2train(:,trainlabel==i-1),b(:,i));
        trainP2 = trainP2 + sum(sum(mat1))-num*sum(log(1+exp(b(:,i)))); 
    end;
    trainP2 = trainP2/ntrain;  
    trainQ2 = sum(sum(H2train.*log(H2train+realmin)+(1-H2train).*log(1-H2train+realmin))); trainQ2 = trainQ2/ntrain;
    TrainLogProb(iter) = trainP0+trainP1+trainP2-trainQ1-trainQ2;
    
    totalP0 = zeros(1,num); totalP1 = zeros(1,num);
    for i = 1:num
        H1samp = H1test>=rand(K1,ntest);
        mat1 = bsxfun(@plus,W1*H1samp,c1);
        totalP0(i) = sum(sum(mat1.*Vtest-log(1+exp(mat1)))); 
        H2samp = H2test>=rand(K2,ntest);
        mat2 = bsxfun(@plus,W2*H2samp,c2);
        totalP1(i) = sum(sum(mat2.*H1test-log(1+exp(mat2)))); 
    end;
    testP0 = mean(totalP0)/ntest; testP1 = mean(totalP1)/ntest;
    testQ1 = sum(sum(H1test.*log(H1test+realmin)+(1-H1test).*log(1-H1test+realmin))); testQ1 = testQ1/ntest;
    testP2 = 0;
    for i = 1:q
        num = sum(testlabel==i-1);
        mat1 = bsxfun(@times,H2test(:,testlabel==i-1),b(:,i));
        testP2 = testP2 + sum(sum(mat1))-num*sum(log(1+exp(b(:,i)))); 
    end;
    testP2 = testP2/ntest;  
    testQ2 = sum(sum(H2test.*log(H2test+realmin)+(1-H2test).*log(1-H2test+realmin))); testQ2 = testQ2/ntest;
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
            figure(3);
            dispims(W1,28,28); title('dictionaries');
            drawnow;
        end;
    end
end;

result.W1 = W1; result.W2 = W2;
result.H1train = H1train; result.H2train = H2train;
result.H1test = H1test; result.H2test = H2test;
result.c1 = c1; result.c2 = c2;
result.b = b;
result.gamma0Train = gamma0Train;
result.gamma0Test = gamma0Test;
result.gamma1Train = gamma1Train;
result.gamma1Test = gamma1Test;
result.gamma2 = gamma2;
result.TrainAcc = TrainAcc; 
result.TestAcc = TestAcc;
result.TotalTime = TotalTime;
result.TrainLogProb = TrainLogProb; 
result.TestLogProb = TestLogProb;


