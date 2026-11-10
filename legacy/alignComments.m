function alignComments(varargin)
% format files in folder (and subfolders)
% v0.96 - 06/02/2024
% written by Tanmay Chhatbar
% alignComments - format all files in active dir
% alignComments({FILE_OR_FOLDER_FULL_OR_RELATIVE_PATH}, ...) - format files in another directory, or a specific file
% (recommended) alignComments(..., 'overwrite', 0) - To format files without overwriting (recommended)
% alignComments(..., 'recursive', 1) - To format files in subfolders as well

%% parse input
p = inputParser;
validbool = @(x) (x==1 || x==0);
validinput = @(x) all(isfolder(x) | isfile(x));
addOptional(p,'directory','./',validinput);
addParameter(p,'recursive',1,validbool);
addParameter(p,'overwrite',1,validbool);
parse(p,varargin{:});

%% do stuff
directory = string(p.Results.directory);  % file/folder to align comments in
for j = 1:length(directory)
    dd = directory(j);
    if isfile(dd)
        filelist(1).name = dd;
        dd='';
    else
        filelist = dir(dd);
    end
    overwrite = p.Results.overwrite;      % overwrite or make _new.m files
    recursive = p.Results.recursive;      % align comments in subfolders too
    % do something
    for i = 1:length(filelist)
        file = filelist(i);
        if endsWith(file.name,'.m')
            add_spaces(fullfile(dd,file.name),overwrite);
        elseif recursive == 1 && isfolder(file.name) && ~any([strcmp(file.name,'..'), strcmp(file.name,'.')])
            alignComments(fullfile(dd, file.name),'recursive',recursive,'overwrite',overwrite);
        end
    end
end
end

%% add whitespaces to lines at idx to match comment at maxidx
function lineout = add_whitespace(line, idx, maxidx)
numspaces = maxidx-idx;
lineout = [line(1:idx-1), repmat(' ',[1 numspaces]), line(idx:end)];
end

%% write lines to file
function writelines(lines, filename)
fileID = fopen(filename,'w');
for i=1:length(lines)
    fprintf(fileID,'%s\n',lines(i));
end
fclose(fileID);
end

%% function add spaces
function add_spaces(filename, overwrite)
% find max idx
lines = readlines(filename);
maxidx = 30;
for j=1:length(lines)
    line = char(lines(j));

    usefulidx = getUsefulIdx(line);
    if isnan(usefulidx)
        continue;
    end
    while 1
        if usefulidx == 1
            usefulidx = 2;
        elseif isspace(line(usefulidx-1)) % discount spaces
            line(usefulidx-1) = [];
            usefulidx = usefulidx-1;
        else
            break
        end
    end
    maxidx = max(maxidx, usefulidx + 1);
end

% align
for j=1:length(lines)
    line = char(lines(j));
    idx = strfind(line,'%');
    if ~isempty(idx)                      % if line has '%'
        % get idx of the "%" that starts a comment
        usefulidx = getUsefulIdx(line);

        % ignore if only comment in line
        if isnan(usefulidx)
            continue
        elseif all(isspace(line(1:usefulidx-1)))
            usefulidx = nan;
        end
        if ~isnan(usefulidx)
            % remove trailing spaces
            while 1
                if isspace(line(usefulidx-1))
                    line(usefulidx-1) = [];
                    usefulidx = usefulidx-1;
                else
                    break
                end
            end
            % add tabs
            lines(j) = string(add_whitespace(line, usefulidx, maxidx));
        end
    end
end

%% save
% remove blank lines in the end
while lines(end) == ""
    lines = lines(1:end-1);
end

% save
if overwrite
    outfile = filename;
else
    filename = char(filename);
    outfile = [filename(1:end-2) '_new.m'];
end
writelines(lines, outfile);
fprintf("Formatted %s\n", outfile);
end

%% functions
function b = ismath(c)
b = any(c=='+-/*'.');
end

function usefulidx = getUsefulIdx(line)
% ignore if "%" in string or character array or transpose
usefulidx = nan;
% cur = 0 is default
% cur = 1 is part of char-array
% cur = 2 is part of string array
cur = 0;

for i = 1:length(line)
    curChar = line(i);
    if i > 1
        prevChar = line(i-1);
    end
    if strcmp(curChar, "'")
        if i == 1                         % first character is "'"
            cur = 1;
        elseif cur == 1
            cur = 0;
        elseif isletter(prevChar) || any(prevChar=='.)'.')
            if cur == 1                   % in character array, end of array
                cur = 0;
            elseif cur == 2               % in string, no change
                0;
            elseif cur == 0               % transpose
                cur = 0;
            end
        elseif isspace(prevChar) || ismath(prevChar) || any(prevChar==',(={'.')
            if cur == 1                   % in character array, end of array
                cur = 0;
            elseif cur == 2               % in string, no change
                0;
            elseif cur == 0               % not in char-array, start of char-array
                cur = 1;
            end
        elseif cur == 0                   % if not part of string
            error("unknown character: " + prevChar + curChar)
        end
    elseif strcmp(curChar, '"')
        if i == 1                         % first character is '"'
            cur = 2;
        elseif cur == 0                   % start of string
            cur = 2;
        elseif cur == 1                   % part of char-array, no change
            0;
        elseif cur == 2                   % part of string, end of string
            cur = 0;
        end
    end
    if cur == 0 && curChar == "%"
        usefulidx = i;
        break
    end
end
end
