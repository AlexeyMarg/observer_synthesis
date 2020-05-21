%Function for calculation of Luenberger observer coefficient
function K=LO_calc(A,B,C,D,eig_d)
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

    %Check of observer existence%
  [m, n]=size(C);
  W_obs=C;
  for i=1:(n-1)
     W_obs=[W_obs; C*A^i]; % Observability matrix 
  end
  if rank(W_obs)~=n
      disp("System isn't observable");
      return;
  else
      disp("System is observable");
  end

    %Matrices transformation to canonical form%
  U=B;
  for i=1:(n-1)
     U=[U (A^i)*B]; % Observability matrix 
  end
  syms s;
  W=C*inv(s*eye(n)-A)*B;
  W=char(W);
  s=tf('s');
  eval(['w = ', W]);
  [num , den]=tfdata(w);
  num=cell2mat(num);
  den=cell2mat(den);
  A_s=zeros(n,n);
  A_s(1:(n-1),2:n)=eye(n-1);
  for i=1:n
      A_s(n,i)=-den(n+2-i);
  end
  B_s=zeros(n, 1);
  B_s(n,1)=1;
  C_s=zeros(1, n);
  for i=1:n
       C_s(1,i)=num(n+2-i);
  end
  U_s=B_s;
  for i=1:(n-1)
     U_s=[U_s (A_s^i)*B_s]; % Observability matrix 
  end
  P=U_s*inv(U);
    % Calculation of desired F=A-KC%
  var=1;
  %calculation of characteristic polynomial
  syms s;
  for i=1:n
     var=var*(s-eig_d(i)); 
  end
  var=sym2poly(collect(var));
  % F construction in canonucal form
  F=zeros(n,n);
  F(1:(n-1),2:n)=eye(n-1);
  for i=1:n
      F(n,i)=-var(n+2-i);
  end
      %K calculation%
  K_s=(A_s-F)*C_s'*inv(C_s*C_s');
  K=inv(P)*K_s;
  disp("Observer matrix")
  K
  clear A_s B_s C_s den eig_d F i K_s m num P U U_s var w W W_obs
end