
function [logpv,logweights]=functionEvaluation(v,parameters,AIS_Samples)
N=size(v,2);
W=parameters{1};
c=parameters{2};
b=parameters{3};
logweights=zeros(1,N);
Samples=AIS_Samples;

% collect
Temps=linspace(0,1,Samples+1);
[h,~]=HgibbsSampler(W*0,b*0,c*0,v*0,1);

Wo=W*0;
bo=b*0;
co=c*0;
for iter=2:Samples+1
    Wt=W*Temps(iter);
    bt=b*Temps(iter);
    ct=c*Temps(iter);
    [h,~]=HgibbsSampler(Wt,bt,ct,v,1,h);
    logweights=logweights+calcNegEnergy(v,h,Wt,bt,ct)-calcNegEnergy(v,h,Wo,bo,co);
    Wo=Wt;
    bo=bt;
    co=ct;
end
logpv=-size(v,1)*log(2)+mean(logweights);
end
        
function [nE]=calcNegEnergy(v,h,W,b,c)
nE=c'*v+dot(v,W*h)+b'*h-sum(log(1+exp(bsxfun(@plus,W*h,c))))-sum(log(1+exp(b)));

end

function [h,pv]=HgibbsSampler(W,b,c,v,gibbsSamples,h)
[~,N]=size(v);
J=numel(b);
if nargin<6;
    h=double(bsxfun(@lt,rand(J,N),sigmoid(b)));
end
for g=1:gibbsSamples
    jset=randperm(J);
    for j=jset
        term1=W(:,j)'*v+b(j);
        term2=calcTerm2(W,c,h,j);
        r=term1+term2;
        h(j,:)=rand(1,N)<sigmoid(r);
    end
end
if nargout>1
    [~,pv]=calcPy(W,h,c);
end
end

function t2=calcTerm2(W,c,h,j)
h(j,:)=0;
T=bsxfun(@plus,W*h,c);
t2down= -sum(log(1+exp(T)));
t2up= -sum(log(1+exp(bsxfun(@plus,T,W(:,j)))));
t2=t2up-t2down;
end

function [lpv,pv]=calcPy(W,h,c)
if nargin<3
    c=zeros(size(W,1),1);
end
pv=(sigmoid(bsxfun(@plus,W*h,c)));
lpv=log(pv);
end
