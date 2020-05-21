% Function of UIO observer synthesis
function [F, T, K, H]=UIO_calc(A,B,C,E,x0, eig_d)
[m,n]=size(C);
[var, r]=size(B);
[var,q]=size(E);
    % Check of desired eigenvalues
if length(eig_d)>length(A) || max(eig_d)>=0
    disp("Wrong desired eigenvalues")
    return;
else
    if length(eig_d)<length(A)
        for i=1:(length(A)-length(eig_d))
            eig_d=[eig_d min(eig_d)];
        end
    end
end

H=E * inv((C*E)'*C*E) * ((C*E)');
T=eye(length(A))-H*C;
A1=T*A;

    % Calculation of desired F=A1-K1C%
var=1;
%calculation of characteristic polynomial
syms s;
for i=1:n
   var=var*(s-eig_d(i)); 
end
var=sym2poly(collect(var));
% F construction in canonical form
F=zeros(n,n);
F(2:n,1:n-1)=eye(n-1);
for i=1:n
    F(i,n)=-var(n+2-i);
end
   %Check of observer existance
if rank(C*E)~=rank(E)
    disp("UIO doesn't exist");
    return;
end

% Check of (C,A1) observability
W0=C;
for i=1:1:(n-1)
    W0=[W0; C*A1^i]; 
 end
if rank(W0)==n
    disp("There is no need in decomposition");
    K1=place(A1, C', eig_d)';
    F=A1-K1*C;
    K=K1+F*H;
    disp("Obtained F eigenvalues:");
    eig(F)
else
    disp("Try to perform decomposition");
    n0=rank(W0);
    i=1;
    var=zeros(1, n);
    while rank(var)==0 && i<=n
        var=W0(i, :);
        i=i+1;
    end
    P=var;
    [var, ~]=size(A);
    while rank(P)<n0
        j=W0(i, :)
        if rank([P; j])>rank(P)
            P=[P; j];
        end
    end
    flag=0;
    while flag==0
        while rank(P)<n
            var=randn([1 n]);
            if rank([P ; var])>rank(P)
                P=[P; var];
            end
        end
        if det(P)~=0
            flag=1;
        else
            P=P(1:n0,:);
        end
    end
    var=P*A1*inv(P);
    A11=var(1:n0,1:n0);
    A22=var((n0+1):n,(n0+1):n);
    var=eig(A22);
    for i=1:length(var)
        if var(i)>0
            disp("UIO doesn't exist");
            return;
        end
    end
    var=C*inv(P);
    C_s=var(1:n0,1:n0);
    eig_d_decomp=eig_d(1, 1:length(A11));
    K1p=place(A11, C_s', eig_d_decomp')';
    K2p=ones(n-n0,m);
    K1=inv(P)*[K1p' K2p']';
    F=A1-K1*C;
    K=K1+F*H;
    disp("Obtained F eigenvalues:");
    eig(F)    
end
clear A1 A11 A22 C_s eig_d eig_d_decomp flag i K1 K1p K2p m n n n0 P q r s var W0   
end
