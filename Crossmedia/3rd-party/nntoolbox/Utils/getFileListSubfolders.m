function [flist filenames] = getFileListSubfolders(root, ext, folders, prefix)
% Returns a cell with list of files with the given extension in the 
% subfolders of the root. Each cell is the result of a folder.
% If folders is given then only those folders will be used.
% filenames contains the barebone list of fullfiles
%
% if the subfolder list is not given take them all
if isempty(folders)
    folders = dir(root);
    folders = folders([folders.isdir]);
    folders(strncmp({folders.name}, '.', 1)) = []; % new, no exceptions
    folders = {folders.name};
end

flist = {};
filenames = {};
for i = 1:length(folders)
    flist{i}.name = folders{i};
    ims = dir(fullfile(root, folders{i}, [prefix, '*.', ext]));
    ims = cellfun(@(x)fullfile(root,folders{i},x),{ims.name},'UniformOutput',false) ;
    flist{i}.files = ims;
    
    filenames = {filenames{:}, ims{:}};
end