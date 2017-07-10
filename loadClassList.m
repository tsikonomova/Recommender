%% Load class list 
%  

function classList = loadClassList()

fid = fopen('classes_ids.txt');

n = 1682;  

classList = cell(n, 1);
for i = 1:n
    line = fgets(fid);
    [idx, className] = strtok(line, ' ');
    classList{i} = strtrim(className);
end
fclose(fid);

end