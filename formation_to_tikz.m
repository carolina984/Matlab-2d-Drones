function formation_to_tikz(cfg, fname)
P   = hex7_positions(cfg.traj.formation_spacing);   % 2 x 7
fid = fopen(fname, 'w');
fprintf(fid, '\\begin{tikzpicture}[x=0.9mm, y=0.9mm]\n');
for i = 1:7
    for j = i+1:7
        d = norm(P(:,i) - P(:,j));
        sty = 'link'; if d >= cfg.range_max, sty = 'nolink'; end
        fprintf(fid, '  \\draw[%s] (%.2f,%.2f) -- (%.2f,%.2f);\n', ...
                sty, P(1,i), P(2,i), P(1,j), P(2,j));
    end
end
for i = 1:7
    fprintf(fid, '  \\node[uav] at (%.2f,%.2f) {\\scriptsize %d};\n', P(1,i), P(2,i), i);
end
fprintf(fid, '\\end{tikzpicture}\n');
fclose(fid);