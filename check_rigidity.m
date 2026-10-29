function out = check_rigidity(P, R_comm, verbose)
%CHECK_RIGIDITY  Is this formation localisable from ranges alone?
%
%   out = CHECK_RIGIDITY(P, R_comm)
%   out = CHECK_RIGIDITY(P, R_comm, true)     % print a report
%
%   P       n x d node positions (d = 2 or 3)
%   R_comm  ranging radius; an edge exists when the distance is <= R_comm
%
%   out.edges        m x 2 edge list
%   out.rank         rank of the rigidity matrix
%   out.rank_needed  d*n - d*(d+1)/2
%   out.rigid        rank == rank_needed  (no continuous deformation)
%   out.redundant    still rigid after removing ANY single edge
%   out.connectivity vertex connectivity (capped at 4)
%   out.glob_rigid   2D only: redundant && connectivity >= 3  (Jackson-Jordan)
%   out.degrees      neighbours per node
%   out.triangles    number of 3-cycles  (loopy-BP cost: fewer is better)
%
%   WHAT THE RIGIDITY MATRIX IS
%   Each edge fixes a distance, so  ||p_i - p_j||^2 = const. Differentiate:
%       (p_i - p_j) . (v_i - v_j) = 0
%   Stack one row per edge and this is  R*v = 0.  R is m x d*n: the row for
%   edge (i,j) holds (p_i - p_j) in node i's d columns and (p_j - p_i) in
%   node j's. Its nullspace is the set of motions the ranges allow.
%
%   Translations and rotations are always allowed -- d*(d+1)/2 of them, so
%   3 in the plane and 6 in space. If the nullspace is exactly that and no
%   bigger, nothing can flex and the framework is infinitesimally rigid.
%
%   RIGID IS NOT ENOUGH
%   A rigid framework can still have DISCRETE ambiguities: reflect part of
%   it and all the distances still match. That is the mirror solution. To
%   rule those out you need GLOBAL rigidity. In the plane the combinatorial
%   test is Jackson and Jordan (2005): 3-connected AND redundantly rigid.
%
%   IN 3D THERE IS NO SUCH SIMPLE TEST. Global rigidity in space is
%   characterised by an equilibrium stress of rank n-d-1 (Connelly 2005;
%   Gortler, Healy and Thurston 2010 showed it is also necessary,
%   generically) -- not by a counting condition. So for the satellite case
%   this function reports rigidity and connectivity, but out.glob_rigid is
%   returned as NaN and you should say so rather than claim the 2D test.
%   [These three references are from memory -- verify before citing.]

if nargin < 3, verbose = false; end
[n, d] = size(P);
assert(d == 2 || d == 3, 'P must be n x 2 or n x 3');

% ---- edges -------------------------------------------------------------
E = zeros(0,2);
for i = 1:n
    for j = i+1:n
        if norm(P(i,:) - P(j,:)) <= R_comm + 1e-9
            E(end+1,:) = [i j];                                   %#ok<AGROW>
        end
    end
end
out.edges = E;
m = size(E,1);

% ---- the rigidity matrix ----------------------------------------------
out.rank        = rigidity_rank(P, E);
out.rank_needed = d*n - d*(d+1)/2;
out.rigid       = (out.rank == out.rank_needed);

% ---- redundant rigidity: drop each edge in turn ------------------------
out.redundant = out.rigid;
if out.rigid
    for e = 1:m
        Em = E([1:e-1, e+1:m], :);
        if rigidity_rank(P, Em) ~= out.rank_needed
            out.redundant = false; break
        end
    end
end

% ---- vertex connectivity (brute force; fine for n <= ~12) --------------
out.connectivity = 0;
for kk = 1:min(4, n-1)
    ok = true;
    C = nchoosek(1:n, kk-1);
    if kk == 1, C = zeros(1,0); end
    for r = 1:size(C,1)
        if ~is_connected(n, E, C(r,:)), ok = false; break; end
    end
    if ok, out.connectivity = kk; else, break; end
end

% ---- global rigidity ---------------------------------------------------
if d == 2
    out.glob_rigid = out.redundant && out.connectivity >= 3;
else
    out.glob_rigid = NaN;      % see the note above: no counting test in 3D
end

% ---- descriptive -------------------------------------------------------
out.degrees = zeros(1,n);
for i = 1:n, out.degrees(i) = sum(E(:,1)==i | E(:,2)==i); end

A = false(n);
for e = 1:m, A(E(e,1),E(e,2)) = true; A(E(e,2),E(e,1)) = true; end
out.triangles = trace(double(A)^3) / 6;
out.tri_per_edge = out.triangles / max(m,1);

if verbose
    fprintf('\n%dD formation, %d nodes, ranging radius %.3g\n', d, n, R_comm);
    fprintf('  edges            %d   (degrees %d to %d)\n', m, min(out.degrees), max(out.degrees));
    fprintf('  rigidity rank    %d  of  %d needed\n', out.rank, out.rank_needed);
    fprintf('  rigid            %d\n', out.rigid);
    fprintf('  redundantly rigid%d\n', out.redundant);
    fprintf('  connectivity     %d\n', out.connectivity);
    if d == 2
        fprintf('  GLOBALLY RIGID   %d', out.glob_rigid);
        if ~out.glob_rigid
            if ~out.rigid, fprintf('   <- flexible: a continuous deformation exists');
            else,          fprintf('   <- rigid but AMBIGUOUS: mirror solutions exist'); end
        end
        fprintf('\n');
    else
        fprintf('  global rigidity  not decidable by counting in 3D (see help)\n');
    end
    fprintf('  triangles        %d   (%.2f per edge; fewer is kinder to loopy BP)\n', ...
            out.triangles, out.tri_per_edge);
end
end


function r = rigidity_rank(P, E)
[n, d] = size(P);
M = zeros(size(E,1), d*n);
for e = 1:size(E,1)
    i = E(e,1); j = E(e,2);
    v = P(i,:) - P(j,:);
    M(e, d*(i-1)+(1:d)) =  v;
    M(e, d*(j-1)+(1:d)) = -v;
end
r = rank(M, 1e-8);
end


function tf = is_connected(n, E, removed)
keep = setdiff(1:n, removed);
if isempty(keep), tf = true; return; end
A = false(n);
for e = 1:size(E,1)
    if ~ismember(E(e,1), removed) && ~ismember(E(e,2), removed)
        A(E(e,1),E(e,2)) = true; A(E(e,2),E(e,1)) = true;
    end
end
seen = false(1,n); stack = keep(1); seen(keep(1)) = true;
while ~isempty(stack)
    v = stack(end); stack(end) = [];
    nb = find(A(v,:) & ~seen);
    seen(nb) = true; stack = [stack nb];                          %#ok<AGROW>
end
tf = all(seen(keep));
end
